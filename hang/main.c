// vkhang: shock2quest's world as the game draws it on the console, cut down a phase at a time, to find what hangs the
// GPU. The game hangs it at random, more often the shorter its world shader, with RBBM_STATUS 0xEEA51110 and the ring
// read to its end. Each phase draws WALLS full-screen walls back to front, every pixel shaded by every wall, and waits
// for each frame; a phase that stops short is where the GPU stopped.

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "harness.h"
#include "shaders.h"
#include "xsdk/log.h"

enum {
	WALLS = 54,
	// Quads across and down a wall: 900 vertices, as the game's larger walls have.
	COLUMNS = 15,
	ROWS = 10,
	WALL_VERTICES = COLUMNS * ROWS * 6,
	// Position, texture, lightmap and atlas coordinates, normal: the game's world vertex.
	VERTEX_FLOATS = 14,
	STRIDE_BYTES = VERTEX_FLOATS * 4,
	TEX_OFFSET = 12,
	LIGHTMAP_OFFSET = 20,
	ATLAS_OFFSET = 28,
	NORMAL_OFFSET = 44,
	ATLAS_WIDTH = 4096,
	ATLAS_HEIGHT = 2048,
	// Each wall's lightmap: a patch of the atlas this many texels on a side.
	PATCH = 64,
	DIFFUSE = 128,
	DIFFUSE_LEVELS = 8,
	TEXEL_BYTES = 4,
	// The fragment block sits past the vertex block, at the device's offset alignment.
	FRAGMENT_BLOCK_OFFSET = 256,
	UNIFORM_BYTES = FRAGMENT_BLOCK_OFFSET + 32,
	// The game's world blocks (engine/src/materials/lightmap_material.rs, as xost lays them out): the vertex block's
	// worldWave, world, view and projection, and the fragment block's members at these byte offsets.
	GAME_VERTEX_BYTES = 208,
	GAME_FRAGMENT_BYTES = 992,
	GAME_FRAGMENT_OFFSET = 256,
	GAME_UNIFORM_BYTES = GAME_FRAGMENT_OFFSET + GAME_FRAGMENT_BYTES,
	G_LOOP_SCALE = 0,
	G_FOG_EYE_DENSITY = 256,
	G_BUMP_STRENGTH = 272,
	G_AMBIENT = 352,
	G_LIGHTMAP_INTENSITY = 364,
	G_LAYER_COLOR_ALPHA = 384,
	// Its samplers' bindings: pointShadowAtlas 2, shineEnvironment 3 (a cube), shineMask 4, shineRamp 5, shineVeins 6,
	// texture1 (the lightmap) 7, texture2 (the diffuse) 8.
	GAME_BINDINGS = 9,
	GAME_CUBE_BINDING = 3,
	GAME_LIGHTMAP_BINDING = 7,
	GAME_DIFFUSE_BINDING = 8,
	FRAMES_A_STEP = 60,
	STEPS = 3,
	FENCE_TIMEOUT_NS = 1000000000,
	// A frame's wait: long enough that a slow frame isn't taken for a hang.
	FRAME_TIMEOUT_MS = 30000,
	// Frames each phase logs the time of, from its first.
	TIMED_FRAMES = 5,
};

typedef struct {
	const char *name;
	const uint32_t *fragment_spirv;
	uint32_t fragment_bytes;
	uint32_t walls;
	// The game's own world modules, from its console bundle, bound as the engine binds them, with this vertex module
	// (linked to its fragment module, so each has its own).
	bool game;
	const uint32_t *vertex_spirv;
	uint32_t vertex_bytes;
	// Which walls: ON_SCREEN, NEAR_FAR (each crossing the near and far planes) or HUGE (64 screens across).
	uint32_t geometry;
	// Each draw as the game issues it: its own vertex buffer binding and descriptor set, alternating textures.
	bool alternate;
} Phase;

enum { ON_SCREEN, NEAR_FAR, HUGE, GEOMETRIES };

// One full-screen wall each, to time a layer of each shader: the game's world shader took 68 ms a layer on the
// console, and 54 layers 3.6 s, a frame the harness's 1 s fence wait took for a hang. The loops are the game's
// spotlight loop's shape, counted by a uniform (a 255-pass hardware loop left on its condition) and by a constant
// (unrolled).
#define GAME(name, v, V, geometry, alternate) \
	{ "the game's world shader" name, hang_game_##v##_frag_spv_spirv, HANG_GAME_##V##_FRAG_SPV_SPIRV_BYTES, WALLS, true, \
		hang_game_##v##_vert_spv_spirv, HANG_GAME_##V##_VERT_SPV_SPIRV_BYTES, geometry, alternate }
// The game hangs the GPU the sooner the faster its world shader, with the primitive assembler busy and the scan
// converter idle; the walls on screen never did. These walls make the clipper work: across the near and far planes,
// and far past the screen's edges.
static const Phase phases[] = {
	GAME(" less the spotlight loop, 54 walls through the eye", noloop, NOLOOP, NEAR_FAR, false),
	GAME(" less the spotlight loop, 54 walls through the eye, each its own set", noloop, NOLOOP, NEAR_FAR, true),
	GAME(", 54 walls through the eye, each its own set", world, WORLD, NEAR_FAR, true),
};

static const VkVertexInputBindingDescription binding = { 0, STRIDE_BYTES, VK_VERTEX_INPUT_RATE_VERTEX };
static const VkVertexInputAttributeDescription attributes[] = {
	{ .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = 0 },
	{ .location = 1, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT, .offset = TEX_OFFSET },
	{ .location = 2, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT, .offset = LIGHTMAP_OFFSET },
	{ .location = 3, .binding = 0, .format = VK_FORMAT_R32G32B32A32_SFLOAT, .offset = ATLAS_OFFSET },
	{ .location = 4, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = NORMAL_OFFSET },
};

typedef struct {
	VkImage depth;
	VkDeviceMemory depth_memory;
	VkImageView depth_view;
	VkRenderPass render_pass;
	VkFramebuffer framebuffers[HARNESS_MAX_IMAGES];
	VkDescriptorSetLayout set_layout;
	VkPipelineLayout layout;
	VkBuffer vertices;
	VkDeviceMemory vertex_memory;
	VkBuffer uniforms;
	VkDeviceMemory uniform_memory;
	VkImage atlas;
	VkDeviceMemory atlas_memory;
	VkImageView atlas_view;
	VkImage diffuse;
	VkDeviceMemory diffuse_memory;
	VkImageView diffuse_view;
	VkSampler atlas_sampler;
	VkSampler diffuse_sampler;
	VkDescriptorPool pool;
	VkDescriptorSet set;
	// The game's: its layout, blocks, the 1-texel white images the engine binds to a unit with no texture, and a set.
	VkDescriptorSetLayout game_set_layout;
	VkPipelineLayout game_layout;
	VkBuffer game_uniforms;
	VkDeviceMemory game_uniform_memory;
	VkImage white;
	VkDeviceMemory white_memory;
	VkImageView white_view;
	VkImage white_cube;
	VkDeviceMemory white_cube_memory;
	VkImageView white_cube_view;
	VkDescriptorPool game_pool;
	VkDescriptorSet game_set;
	// game_set with the atlas, 4096x2048 and one level, as the diffuse: another texture to switch to.
	VkDescriptorSet game_set2;
} Scene;

void check(const char *name, bool passed)
{
	xsdk_log_write("%s: %s", passed ? "PASS" : "FAIL", name);
}

static uint64_t now_us(void)
{
	struct timespec now;
	(void)clock_gettime(CLOCK_MONOTONIC, &now);
	return (uint64_t)now.tv_sec * 1000000u + (uint64_t)now.tv_nsec / 1000u;
}

// The GPU's registers, which titles can read at 0x7FC80000 (xsdk:docs/HARDWARE.md, "Titles can read the GPU
// registers"): RBBM_STATUS and the ring's read and write pointers, to see where a hang stopped.
static void log_gpu_state(const char *when)
{
	volatile const uint32_t *const registers = (volatile const uint32_t *)0x7FC80000u;
	xsdk_log_write("vkhang: %s: RBBM_STATUS %08X, ring read 0x%X write 0x%X", when,
		(unsigned)registers[0x1740 / 4], (unsigned)registers[0x710 / 4], (unsigned)registers[0x714 / 4]);
}

// One frame: acquire, submit, wait up to FRAME_TIMEOUT_MS, present. Returns the microseconds from the submit to its
// fence, or 0 if any step failed.
static uint32_t frame(const Harness *harness, const VkCommandBuffer *commands)
{
	uint32_t index = 0;
	const uint64_t timeout_ns = (uint64_t)FRAME_TIMEOUT_MS * 1000000u;
	if (!harness_ok("vkAcquireNextImageKHR",
			vkAcquireNextImageKHR(harness->device, harness->swapchain, timeout_ns, VK_NULL_HANDLE, VK_NULL_HANDLE, &index)))
		return 0;
	const VkSubmitInfo submit = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.commandBufferCount = 1,
		.pCommandBuffers = &commands[index],
	};
	const uint64_t start = now_us();
	if (!harness_ok("vkQueueSubmit", vkQueueSubmit(harness->queue, 1, &submit, harness->fence)))
		return 0;
	if (!harness_ok("vkWaitForFences", vkWaitForFences(harness->device, 1, &harness->fence, VK_TRUE, timeout_ns))) {
		log_gpu_state("the frame didn't finish");
		return 0;
	}
	const uint64_t took = now_us() - start;
	(void)vkResetFences(harness->device, 1, &harness->fence);
	const VkPresentInfoKHR present = {
		.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
		.swapchainCount = 1,
		.pSwapchains = &harness->swapchain,
		.pImageIndices = &index,
	};
	if (!harness_ok("vkQueuePresentKHR", vkQueuePresentKHR(harness->queue, &present)))
		return 0;
	return took ? (uint32_t)took : 1;
}

// Wall k covers the screen at depth 0.9 down to 0.1, so each passes the depth test over the one before; its texture
// repeats 4 times across, and its lightmap is the kth patch of the atlas.
static void fill_vertices(float *out)
{
	for (uint32_t g = 0; g < GEOMETRIES; g++)
	for (uint32_t wall = 0; wall < WALLS; wall++) {
		const float z = 0.9f - 0.8f * (float)wall / (float)WALLS;
		const float scale = g == HUGE ? 64.0f : 1.0f;
		const float atlas[4] = { (float)((wall % 32) * 2 * PATCH) / ATLAS_WIDTH,
			(float)((wall / 32) * 2 * PATCH) / ATLAS_HEIGHT, (float)PATCH / ATLAS_WIDTH, (float)PATCH / ATLAS_HEIGHT };
		for (uint32_t row = 0; row < ROWS; row++) {
			for (uint32_t column = 0; column < COLUMNS; column++) {
				static const uint32_t corners[6][2] = { { 0, 0 }, { 1, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 }, { 0, 1 } };
				for (uint32_t c = 0; c < 6; c++) {
					const float u = (float)(column + corners[c][0]) / COLUMNS;
					const float v = (float)(row + corners[c][1]) / ROWS;
					// Across the near and far planes: z from -0.6 to 1.4 down the wall.
					const float vz = g == NEAR_FAR ? (v - 0.3f) * 2.0f + (z - 0.5f) * 0.1f : z;
					const float vertex[VERTEX_FLOATS] = { scale * (-1.0f + 2.0f * u), scale * (-1.0f + 2.0f * v), vz, 4.0f * u, 4.0f * v, u,
						v, atlas[0], atlas[1], atlas[2], atlas[3], 0.0f, 0.0f, 1.0f };
					memcpy(out, vertex, sizeof vertex);
					out += VERTEX_FLOATS;
				}
			}
		}
	}
}

// The atlas's texels, then each diffuse level's: gradients, so neighbouring samples differ.
static uint8_t *make_texels(uint32_t *size_bytes)
{
	uint32_t size = ATLAS_WIDTH * ATLAS_HEIGHT * TEXEL_BYTES;
	for (uint32_t level = 0; level < DIFFUSE_LEVELS; level++)
		size += (DIFFUSE >> level) * (DIFFUSE >> level) * TEXEL_BYTES;
	const uint32_t white_offset = size;
	size += 7 * TEXEL_BYTES;
	uint8_t *const texels = malloc(size);
	if (!texels)
		return NULL;
	uint8_t *at = texels;
	for (uint32_t y = 0; y < ATLAS_HEIGHT; y++) {
		for (uint32_t x = 0; x < ATLAS_WIDTH; x++, at += TEXEL_BYTES)
			at[0] = (uint8_t)(64 + x % 192), at[1] = (uint8_t)(64 + y % 192), at[2] = 128, at[3] = 255;
	}
	for (uint32_t level = 0; level < DIFFUSE_LEVELS; level++) {
		const uint32_t side = DIFFUSE >> level;
		for (uint32_t y = 0; y < side; y++) {
			for (uint32_t x = 0; x < side; x++, at += TEXEL_BYTES)
				at[0] = (uint8_t)(x * 255 / side), at[1] = (uint8_t)(y * 255 / side), at[2] = (uint8_t)(level * 32),
				at[3] = 255;
		}
	}
	memset(texels + white_offset, 255, 7 * TEXEL_BYTES);
	*size_bytes = size;
	return texels;
}

static VkResult create_image(const Harness *harness, uint32_t width, uint32_t height, uint32_t levels, bool cube,
	VkImage *image, VkDeviceMemory *memory, VkImageView *view)
{
	const VkImageCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.flags = cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = VK_FORMAT_R8G8B8A8_UNORM,
		.extent = { width, height, 1 },
		.mipLevels = levels,
		.arrayLayers = cube ? 6 : 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	};
	VkResult result = vkCreateImage(harness->device, &info, NULL, image);
	if (result != VK_SUCCESS)
		return result;
	VkMemoryRequirements requirements;
	vkGetImageMemoryRequirements(harness->device, *image, &requirements);
	const VkMemoryAllocateInfo allocate_info = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize = requirements.size,
	};
	result = vkAllocateMemory(harness->device, &allocate_info, NULL, memory);
	if (result == VK_SUCCESS)
		result = vkBindImageMemory(harness->device, *image, *memory, 0);
	const VkImageViewCreateInfo view_info = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = *image,
		.viewType = cube ? VK_IMAGE_VIEW_TYPE_CUBE : VK_IMAGE_VIEW_TYPE_2D,
		.format = VK_FORMAT_R8G8B8A8_UNORM,
		.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, levels, 0, cube ? 6 : 1 },
	};
	if (result == VK_SUCCESS)
		result = vkCreateImageView(harness->device, &view_info, NULL, view);
	return result;
}

static void transition_layers(
	VkCommandBuffer commands, VkImage image, uint32_t levels, uint32_t layers, VkImageLayout from, VkImageLayout to)
{
	const VkImageMemoryBarrier barrier = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.srcAccessMask = from == VK_IMAGE_LAYOUT_UNDEFINED ? 0 : VK_ACCESS_TRANSFER_WRITE_BIT,
		.dstAccessMask =
			to == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL ? VK_ACCESS_TRANSFER_WRITE_BIT : VK_ACCESS_SHADER_READ_BIT,
		.oldLayout = from,
		.newLayout = to,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = image,
		.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, levels, 0, layers },
	};
	vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0,
		NULL, 1, &barrier);
}

static void transition(VkCommandBuffer commands, VkImage image, uint32_t levels, VkImageLayout from, VkImageLayout to)
{
	transition_layers(commands, image, levels, 1, from, to);
}

// Copies the atlas and every diffuse level out of one staging buffer.
static bool upload(const Harness *harness, const Scene *scene, VkBuffer staging)
{
	const VkCommandBufferAllocateInfo allocate_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = harness->pool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1,
	};
	VkCommandBuffer commands = VK_NULL_HANDLE;
	if (!harness_ok("upload command buffer", vkAllocateCommandBuffers(harness->device, &allocate_info, &commands)))
		return false;
	const VkCommandBufferBeginInfo begin = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	(void)vkBeginCommandBuffer(commands, &begin);
	transition(commands, scene->atlas, 1, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
	const VkBufferImageCopy atlas_region = {
		.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
		.imageExtent = { ATLAS_WIDTH, ATLAS_HEIGHT, 1 },
	};
	vkCmdCopyBufferToImage(
		commands, staging, scene->atlas, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &atlas_region);
	transition(commands, scene->atlas, 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	transition(
		commands, scene->diffuse, DIFFUSE_LEVELS, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
	VkBufferImageCopy levels[DIFFUSE_LEVELS];
	uint32_t offset = ATLAS_WIDTH * ATLAS_HEIGHT * TEXEL_BYTES;
	for (uint32_t level = 0; level < DIFFUSE_LEVELS; level++) {
		const uint32_t side = DIFFUSE >> level;
		levels[level] = (VkBufferImageCopy){
			.bufferOffset = offset,
			.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1 },
			.imageExtent = { side, side, 1 },
		};
		offset += side * side * TEXEL_BYTES;
	}
	vkCmdCopyBufferToImage(
		commands, staging, scene->diffuse, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, DIFFUSE_LEVELS, levels);
	transition(commands, scene->diffuse, DIFFUSE_LEVELS, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	// The white texels come last: one for the 2D image, six for the cube's faces.
	transition(commands, scene->white, 1, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
	const VkBufferImageCopy white_region = {
		.bufferOffset = offset,
		.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
		.imageExtent = { 1, 1, 1 },
	};
	vkCmdCopyBufferToImage(commands, staging, scene->white, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &white_region);
	transition(commands, scene->white, 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	transition_layers(
		commands, scene->white_cube, 1, 6, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
	const VkBufferImageCopy cube_region = {
		.bufferOffset = offset + TEXEL_BYTES,
		.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 6 },
		.imageExtent = { 1, 1, 1 },
	};
	vkCmdCopyBufferToImage(
		commands, staging, scene->white_cube, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cube_region);
	transition_layers(commands, scene->white_cube, 1, 6, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	bool ok = harness_ok("upload recording", vkEndCommandBuffer(commands));
	const VkSubmitInfo submit = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &commands
	};
	ok = ok && harness_ok("upload submit", vkQueueSubmit(harness->queue, 1, &submit, harness->fence));
	ok = ok &&
		harness_ok("upload wait", vkWaitForFences(harness->device, 1, &harness->fence, VK_TRUE, FENCE_TIMEOUT_NS));
	(void)vkResetFences(harness->device, 1, &harness->fence);
	vkFreeCommandBuffers(harness->device, harness->pool, 1, &commands);
	return ok;
}

static VkResult create_layout(const Harness *harness, Scene *scene)
{
	const VkDescriptorSetLayoutBinding bindings[] = {
		{ 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, NULL },
		{ 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL },
		{ 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL },
		{ 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL },
	};
	const VkDescriptorSetLayoutCreateInfo set_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 4,
		.pBindings = bindings,
	};
	VkResult result = vkCreateDescriptorSetLayout(harness->device, &set_info, NULL, &scene->set_layout);
	const VkPipelineLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &scene->set_layout,
	};
	if (result == VK_SUCCESS)
		result = vkCreatePipelineLayout(harness->device, &layout_info, NULL, &scene->layout);
	return result;
}

static VkResult create_descriptors(const Harness *harness, Scene *scene)
{
	const VkSamplerCreateInfo atlas_info = {
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.magFilter = VK_FILTER_LINEAR,
		.minFilter = VK_FILTER_LINEAR,
		.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
		.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
	};
	VkResult result = vkCreateSampler(harness->device, &atlas_info, NULL, &scene->atlas_sampler);
	const VkSamplerCreateInfo diffuse_info = {
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.magFilter = VK_FILTER_LINEAR,
		.minFilter = VK_FILTER_LINEAR,
		.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
		.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.maxLod = DIFFUSE_LEVELS,
	};
	if (result == VK_SUCCESS)
		result = vkCreateSampler(harness->device, &diffuse_info, NULL, &scene->diffuse_sampler);
	const VkDescriptorPoolSize sizes[] = {
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2 },
		{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2 },
	};
	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 1,
		.poolSizeCount = 2,
		.pPoolSizes = sizes,
	};
	if (result == VK_SUCCESS)
		result = vkCreateDescriptorPool(harness->device, &pool_info, NULL, &scene->pool);
	const VkDescriptorSetAllocateInfo allocate_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = scene->pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &scene->set_layout,
	};
	if (result == VK_SUCCESS)
		result = vkAllocateDescriptorSets(harness->device, &allocate_info, &scene->set);
	if (result != VK_SUCCESS)
		return result;
	const VkDescriptorBufferInfo blocks[2] = {
		{ scene->uniforms, 0, 64 },
		{ scene->uniforms, FRAGMENT_BLOCK_OFFSET, 32 },
	};
	const VkDescriptorImageInfo images[2] = {
		{ scene->atlas_sampler, scene->atlas_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
		{ scene->diffuse_sampler, scene->diffuse_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
	};
	VkWriteDescriptorSet writes[4];
	for (uint32_t i = 0; i < 4; i++)
		writes[i] = (VkWriteDescriptorSet){
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = scene->set,
			.dstBinding = i,
			.descriptorCount = 1,
			.descriptorType = i < 2 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.pBufferInfo = i < 2 ? &blocks[i] : NULL,
			.pImageInfo = i < 2 ? NULL : &images[i - 2],
		};
	vkUpdateDescriptorSets(harness->device, 4, writes, 0, NULL);
	return VK_SUCCESS;
}

// The engine's layout for a program: binding 0 the vertex block, 1 the fragment block, then a sampler each.
static VkResult create_game_layout(const Harness *harness, Scene *scene)
{
	VkDescriptorSetLayoutBinding bindings[GAME_BINDINGS];
	for (uint32_t i = 0; i < GAME_BINDINGS; i++)
		bindings[i] = (VkDescriptorSetLayoutBinding){
			.binding = i,
			.descriptorType = i < 2 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = 1,
			.stageFlags = i == 0 ? VK_SHADER_STAGE_VERTEX_BIT : VK_SHADER_STAGE_FRAGMENT_BIT,
		};
	const VkDescriptorSetLayoutCreateInfo set_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = GAME_BINDINGS,
		.pBindings = bindings,
	};
	VkResult result = vkCreateDescriptorSetLayout(harness->device, &set_info, NULL, &scene->game_set_layout);
	// The engine's shader::to_vulkan_glsl adds a push block to every vertex shader: flip_y, which it pushes as -1.
	const VkPushConstantRange push = { VK_SHADER_STAGE_VERTEX_BIT, 0, 4 };
	const VkPipelineLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &scene->game_set_layout,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &push,
	};
	if (result == VK_SUCCESS)
		result = vkCreatePipelineLayout(harness->device, &layout_info, NULL, &scene->game_layout);
	return result;
}

static void put_float(float *block, uint32_t offset, float value)
{
	block[offset / 4] = value;
}

// The game's blocks: the identity for world, view and projection; the loop scale 1 (vk_backend writes it so), the
// lightmap at full, a dim ambient, an opaque white layer colour, a little fog and bump; everything else 0: unlit,
// layer, alpha test, shine, point shadows and the spotlights off.
static VkResult create_game_uniforms(const Harness *harness, Scene *scene)
{
	float uniforms[GAME_UNIFORM_BYTES / 4] = { 0 };
	for (uint32_t matrix = 0; matrix < 3; matrix++) {
		for (uint32_t i = 0; i < 4; i++)
			uniforms[4 + matrix * 16 + i * 5] = 1.0f;
	}
	// The projection gives w = z and depth z / 2: the near-and-far walls run from behind the eye (w -0.6) to in front
	// of it (w 1.4), the case the clipper splits triangles at w = 0 for.
	float *const projection = &uniforms[4 + 2 * 16];
	projection[2 * 4 + 2] = 0.5f;
	projection[2 * 4 + 3] = 1.0f;
	projection[3 * 4 + 3] = 0.0f;
	float *const fragment = &uniforms[GAME_FRAGMENT_OFFSET / 4];
	put_float(fragment, G_LOOP_SCALE, 1.0f);
	put_float(fragment, G_FOG_EYE_DENSITY + 12, 0.04f);
	put_float(fragment, G_BUMP_STRENGTH, 0.7f);
	for (uint32_t i = 0; i < 3; i++)
		put_float(fragment, G_AMBIENT + i * 4, 0.05f);
	put_float(fragment, G_LIGHTMAP_INTENSITY, 1.0f);
	for (uint32_t i = 0; i < 4; i++)
		put_float(fragment, G_LAYER_COLOR_ALPHA + i * 4, 1.0f);
	return harness_buffer(harness, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, uniforms, sizeof uniforms,
		&scene->game_uniforms, &scene->game_uniform_memory);
}

static VkResult create_game_descriptors(const Harness *harness, Scene *scene)
{
	const VkDescriptorPoolSize sizes[] = {
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4 },
		{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2 * (GAME_BINDINGS - 2) },
	};
	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 2,
		.poolSizeCount = 2,
		.pPoolSizes = sizes,
	};
	VkResult result = vkCreateDescriptorPool(harness->device, &pool_info, NULL, &scene->game_pool);
	const VkDescriptorSetAllocateInfo allocate_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = scene->game_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &scene->game_set_layout,
	};
	if (result == VK_SUCCESS)
		result = vkAllocateDescriptorSets(harness->device, &allocate_info, &scene->game_set);
	if (result == VK_SUCCESS)
		result = vkAllocateDescriptorSets(harness->device, &allocate_info, &scene->game_set2);
	if (result != VK_SUCCESS)
		return result;
	const VkDescriptorBufferInfo blocks[2] = {
		{ scene->game_uniforms, 0, GAME_VERTEX_BYTES },
		{ scene->game_uniforms, GAME_FRAGMENT_OFFSET, GAME_FRAGMENT_BYTES },
	};
	VkDescriptorImageInfo images[GAME_BINDINGS];
	VkWriteDescriptorSet writes[GAME_BINDINGS];
	for (uint32_t set = 0; set < 2; set++)
	for (uint32_t i = 0; i < GAME_BINDINGS; i++) {
		VkImageView view = scene->white_view;
		VkSampler sampler = scene->atlas_sampler;
		if (i == GAME_CUBE_BINDING)
			view = scene->white_cube_view;
		else if (i == GAME_LIGHTMAP_BINDING)
			view = scene->atlas_view;
		else if (i == GAME_DIFFUSE_BINDING)
			view = set ? scene->atlas_view : scene->diffuse_view, sampler = scene->diffuse_sampler;
		images[i] = (VkDescriptorImageInfo){ sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
		writes[i] = (VkWriteDescriptorSet){
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = set ? scene->game_set2 : scene->game_set,
			.dstBinding = i,
			.descriptorCount = 1,
			.descriptorType = i < 2 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.pBufferInfo = i < 2 ? &blocks[i] : NULL,
			.pImageInfo = i < 2 ? NULL : &images[i],
		};
		if (i == GAME_BINDINGS - 1)
			vkUpdateDescriptorSets(harness->device, GAME_BINDINGS, writes, 0, NULL);
	}
	return VK_SUCCESS;
}

static bool build(const Harness *harness, Scene *scene)
{
	bool ok = harness_ok("depth image",
		harness_depth_image(harness, &scene->depth, &scene->depth_memory, &scene->depth_view));
	ok = ok &&
		harness_ok("render pass",
			harness_depth_render_pass(harness, scene->depth_view, &scene->render_pass, scene->framebuffers));
	ok = ok && harness_ok("layout", create_layout(harness, scene));

	const uint32_t vertex_bytes = GEOMETRIES * WALLS * WALL_VERTICES * STRIDE_BYTES;
	float *const vertices = malloc(vertex_bytes);
	ok = ok && vertices;
	if (ok)
		fill_vertices(vertices);
	ok = ok &&
		harness_ok("vertex buffer",
			harness_buffer(harness, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertices, vertex_bytes, &scene->vertices,
				&scene->vertex_memory));
	free(vertices);

	// The vertex block: the identity. The fragment block: a dim ambient, the lightmap at full, no alpha test, a
	// little fog, and the loops' scale 1, as the game's shockLoopScale.
	float uniforms[UNIFORM_BYTES / 4] = { 0 };
	for (uint32_t i = 0; i < 4; i++)
		uniforms[i * 5] = 1.0f;
	const float fragment[8] = { 0.05f, 0.05f, 0.08f, 1.0f, 1.0f, 0.0f, 0.04f, 1.0f };
	memcpy(&uniforms[FRAGMENT_BLOCK_OFFSET / 4], fragment, sizeof fragment);
	ok = ok &&
		harness_ok("uniform buffer",
			harness_buffer(harness, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, uniforms, sizeof uniforms, &scene->uniforms,
				&scene->uniform_memory));

	ok = ok &&
		harness_ok("atlas",
			create_image(harness, ATLAS_WIDTH, ATLAS_HEIGHT, 1, false, &scene->atlas, &scene->atlas_memory,
				&scene->atlas_view));
	ok = ok &&
		harness_ok("diffuse",
			create_image(harness, DIFFUSE, DIFFUSE, DIFFUSE_LEVELS, false, &scene->diffuse, &scene->diffuse_memory,
				&scene->diffuse_view));
	ok = ok &&
		harness_ok("white image",
			create_image(harness, 1, 1, 1, false, &scene->white, &scene->white_memory, &scene->white_view));
	ok = ok &&
		harness_ok("white cube",
			create_image(
				harness, 1, 1, 1, true, &scene->white_cube, &scene->white_cube_memory, &scene->white_cube_view));
	uint32_t texel_bytes = 0;
	uint8_t *const texels = ok ? make_texels(&texel_bytes) : NULL;
	VkBuffer staging = VK_NULL_HANDLE;
	VkDeviceMemory staging_memory = VK_NULL_HANDLE;
	ok = ok && texels &&
		harness_ok("staging buffer",
			harness_buffer(harness, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, texels, texel_bytes, &staging, &staging_memory));
	free(texels);
	ok = ok && upload(harness, scene, staging);
	if (staging) {
		vkDestroyBuffer(harness->device, staging, NULL);
		vkFreeMemory(harness->device, staging_memory, NULL);
	}
	ok = ok && harness_ok("descriptors", create_descriptors(harness, scene));
	ok = ok && harness_ok("game layout", create_game_layout(harness, scene));
	ok = ok && harness_ok("game uniforms", create_game_uniforms(harness, scene));
	return ok && harness_ok("game descriptors", create_game_descriptors(harness, scene));
}

static VkResult create_pipeline(const Harness *harness, const Scene *scene, const Phase *phase, VkPipeline *pipeline)
{
	static const VkPipelineDepthStencilStateCreateInfo depth = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_LESS,
	};
	const HarnessPipeline spec = {
		.vertex_spirv = phase->game ? phase->vertex_spirv : hang_world_vert_spirv,
		.vertex_bytes = phase->game ? phase->vertex_bytes : HANG_WORLD_VERT_SPIRV_BYTES,
		.fragment_spirv = phase->fragment_spirv,
		.fragment_bytes = phase->fragment_bytes,
		.bindings = &binding,
		.binding_count = 1,
		.attributes = attributes,
		.attribute_count = sizeof attributes / sizeof attributes[0],
		.layout = phase->game ? scene->game_layout : scene->layout,
		.depth_stencil = &depth,
	};
	return harness_pipeline(harness, &spec, scene->render_pass, pipeline);
}

static VkResult record(const Harness *harness, const Scene *scene, const Phase *phase, VkPipeline pipeline,
	VkCommandBuffer *commands)
{
	static const VkClearValue clears[2] = {
		{ .color = { .float32 = { 0.0f, 0.0f, 0.0f, 1.0f } } },
		{ .depthStencil = { 1.0f, 0 } },
	};
	VkResult result = harness_command_buffers(harness, commands);
	for (uint32_t i = 0; i < harness->image_count && result == VK_SUCCESS; i++) {
		const VkRenderPassBeginInfo pass = {
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = scene->render_pass,
			.framebuffer = scene->framebuffers[i],
			.renderArea = { { 0, 0 }, { HARNESS_WIDTH, HARNESS_HEIGHT } },
			.clearValueCount = 2,
			.pClearValues = clears,
		};
		const VkDeviceSize offset = 0;
		vkCmdBeginRenderPass(commands[i], &pass, VK_SUBPASS_CONTENTS_INLINE);
		vkCmdBindPipeline(commands[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
		vkCmdBindVertexBuffers(commands[i], 0, 1, &scene->vertices, &offset);
		vkCmdBindDescriptorSets(commands[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
			phase->game ? scene->game_layout : scene->layout, 0, 1, phase->game ? &scene->game_set : &scene->set, 0,
			NULL);
		for (uint32_t wall = 0; wall < phase->walls; wall++) {
			if (phase->alternate) {
				const VkDeviceSize wall_offset =
					(VkDeviceSize)(phase->geometry * WALLS + wall) * WALL_VERTICES * STRIDE_BYTES;
				vkCmdBindVertexBuffers(commands[i], 0, 1, &scene->vertices, &wall_offset);
				vkCmdBindDescriptorSets(commands[i], VK_PIPELINE_BIND_POINT_GRAPHICS, scene->game_layout, 0, 1,
					wall % 2 ? &scene->game_set2 : &scene->game_set, 0, NULL);
			}
			if (phase->game) {
				static const float flip_y = -1.0f;
				vkCmdPushConstants(
					commands[i], scene->game_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof flip_y, &flip_y);
			}
			vkCmdDraw(commands[i], WALL_VERTICES, 1,
				phase->alternate ? 0 : (phase->geometry * WALLS + wall) * WALL_VERTICES, 0);
		}
		vkCmdEndRenderPass(commands[i]);
		result = vkEndCommandBuffer(commands[i]);
	}
	return result;
}

// Whether the phase ran all its frames; the GPU is gone if not.
static bool run(const Harness *harness, const Scene *scene, const Phase *phase)
{
	xsdk_log_write("vkhang: phase \"%s\"", phase->name);
	VkPipeline pipeline = VK_NULL_HANDLE;
	VkCommandBuffer commands[HARNESS_MAX_IMAGES] = { VK_NULL_HANDLE };
	if (!harness_ok("pipeline", create_pipeline(harness, scene, phase, &pipeline)) ||
		!harness_ok("recording", record(harness, scene, phase, pipeline, commands)))
		return false;
	uint32_t total = 0;
	bool whole = true;
	for (uint32_t step = 0; step < STEPS && whole; step++) {
		uint64_t step_us = 0;
		for (uint32_t i = 0; i < FRAMES_A_STEP && whole; i++) {
			const uint32_t took = frame(harness, commands);
			whole = took != 0;
			if (total < TIMED_FRAMES)
				xsdk_log_write("vkhang: \"%s\": frame %u took %u us", phase->name, (unsigned)total, (unsigned)took);
			total += whole;
			step_us += took;
		}
		xsdk_log_write("vkhang: \"%s\": %u frames, %u us a frame", phase->name, (unsigned)total,
			(unsigned)(step_us / FRAMES_A_STEP));
	}
	check(phase->name, whole);
	if (whole) {
		vkFreeCommandBuffers(harness->device, harness->pool, harness->image_count, commands);
		vkDestroyPipeline(harness->device, pipeline, NULL);
	}
	return whole;
}

int main(void)
{
	xsdk_log_write("vkhang: what in shock2quest's world draws hangs the GPU");
	Harness harness = { .instance = VK_NULL_HANDLE };
	Scene scene = { .render_pass = VK_NULL_HANDLE };
	bool ok = harness_ok("device", harness_create_device(&harness)) &&
		harness_ok("swapchain", harness_create_swapchain(&harness, 2)) && build(&harness, &scene);
	check("vkhang: device, swapchain, textures and buffers made", ok);
	for (uint32_t i = 0; ok && i < sizeof phases / sizeof phases[0]; i++)
		ok = run(&harness, &scene, &phases[i]);
	xsdk_log_write(ok ? "vkhang: every phase ran" : "vkhang: stopped");
	return ok ? 0 : 1;
}
