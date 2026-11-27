// vkreplay: shock2quest's frame as the game issued it, from the draw dump the game writes on the console
// (shock2quest:engine/src/gfx/draw_dump.rs, `dump_draws.txt`, `draws.bin`), drawn again each frame with the game's own
// shaders (its console bundle is linked in), state, vertices and uniform blocks, to find what hangs the GPU without the
// game around it. Textures are gradients at the game's sizes and levels.
//
// Each line of `replay.txt` beside the .xex is a phase, its words what to change: `noblend`, `nodepthwrite`,
// `nodepthtest`, `cull` (back faces), `program=N` (only that program's draws), `draws=A-B` (only those, inclusive),
// `verts=A-B` (only those vertices of each draw, inclusive), `mask=0` (no colour written), `noclear`,
// `nodegen` / `onlydegen` (only the triangles with area / only the zero-area ones), `shader=nobump|nodiscard|neither|plain|noloop|tex|flat|fullnoloop|one|oneif`
// (the world program's fragment shader cut down), `spotcount=N` (its spotlightCount uniform); an empty line or `asis` draws the dump as it is. With no file, one phase as
// it is. Phases run in order and stop at the first that hangs.

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "harness.h"
#include "shock2_shaders.h"
#include "xsdk/log.h"

enum {
	TAG_FRAM = 0x4D415246,
	TAG_CAPB = 0x42504143,
	TAG_CAPE = 0x45504143,
	TAG_CLER = 0x52454C43,
	TAG_PROG = 0x474F5250,
	TAG_DRAW = 0x57415244,
	MAX_PROGRAMS = 16,
	MAX_DRAWS = 512,
	MAX_CLEARS = 16,
	MAX_ATTRIBUTES = 16,
	MAX_SAMPLERS = 16,
	MAX_BINDINGS = 24,
	MAX_IMAGES = 32,
	MAX_PHASES = 32,
	FRAMES_A_STEP = 60,
	STEPS = 3,
	TIMED_FRAMES = 5,
	FRAME_TIMEOUT_MS = 30000,
	UNIFORM_ALIGN = 256,
};

typedef struct {
	bool used;
	VkShaderModule vertex;
	VkShaderModule fragment;
	VkDescriptorSetLayout set_layout;
	VkPipelineLayout layout;
	uint32_t binding_count;
	uint32_t bindings[MAX_BINDINGS][3];
} Program;

typedef struct {
	uint32_t program, topology, count, stride;
	uint32_t attribute_count;
	uint32_t attributes[MAX_ATTRIBUTES][4];
	uint32_t cull[2];
	uint32_t depth_test, depth_write, depth_compare, blend;
	uint32_t blend_factors[4];
	uint32_t color_mask, depth_bias;
	float bias[2];
	uint32_t formats[3];
	float viewport[6];
	int32_t scissor[4];
	const uint8_t *vertices;
	uint32_t vertex_bytes;
	const uint8_t *indices;
	uint32_t index_bytes;
	const uint8_t *blocks[2];
	uint32_t block_bytes[2];
	uint32_t sampler_count;
	uint32_t samplers[MAX_SAMPLERS][7];
	// Made by build().
	VkBuffer vertex_buffer, index_buffer;
	VkDeviceMemory vertex_memory, index_memory;
	VkDeviceSize block_offsets[2];
	VkDescriptorSet set;
	// The draw's triangles split by area: [0] those with some, [1] the ones with none (collinear corners).
	VkBuffer split_buffers[2];
	VkDeviceMemory split_memory[2];
	uint32_t split_counts[2];
} Draw;

typedef struct {
	uint32_t color, depth;
	float rgba[4];
	int32_t rect[4];
} Clear;

// The frame in order: a clear or a draw.
typedef struct {
	bool is_clear;
	uint32_t index;
} Step;

typedef struct {
	uint32_t width, height, levels, layers;
	VkImage image;
	VkDeviceMemory memory;
	VkImageView view;
} Image;

typedef struct {
	bool no_blend, no_depth_write, no_depth_test, cull, no_color, no_clear;
	// Draw only the triangles with area (0), only those without (1), or both (-1).
	int32_t split;
	// 0, or which of `variants` the world program (1) draws with instead of its own fragment module.
	uint32_t shader;
	int32_t program;
	uint32_t first, last;
	// Only these vertices of each draw (inclusive, whole triangles).
	uint32_t vertex_first, vertex_last;
	// spotlightCount for the world program's draws, or -1 for the dump's.
	int32_t spot_count;
	char name[96];
} Phase;

typedef struct {
	Program programs[MAX_PROGRAMS];
	Draw draws[MAX_DRAWS];
	uint32_t draw_count;
	Clear clears[MAX_CLEARS];
	uint32_t clear_count;
	Step steps[MAX_DRAWS + MAX_CLEARS];
	uint32_t step_count;
	uint32_t frame_size[2];
	Image images[MAX_IMAGES];
	uint32_t image_count;
	VkSampler samplers[3][2];
	VkBuffer zero;
	VkDeviceMemory zero_memory;
	VkBuffer uniforms;
	VkDeviceMemory uniform_memory;
	VkDescriptorPool pool;
	VkImage depth;
	VkDeviceMemory depth_memory;
	VkImageView depth_view;
	VkRenderPass render_pass;
	VkFramebuffer framebuffers[HARNESS_MAX_IMAGES];
} Replay;

static Replay replay;

// The world program's fragment shader cut down (hang/replay_shaders.sh): each links to the game's own vertex module.
static const struct {
	const char *name;
	const uint32_t *spirv;
	uint32_t bytes;
} variants[] = {
	{ "", NULL, 0 },
	{ "nobump", shock_nobump_frag_spv_spirv, SHOCK_NOBUMP_FRAG_SPV_SPIRV_BYTES },
	{ "nodiscard", shock_nodiscard_frag_spv_spirv, SHOCK_NODISCARD_FRAG_SPV_SPIRV_BYTES },
	{ "neither", shock_neither_frag_spv_spirv, SHOCK_NEITHER_FRAG_SPV_SPIRV_BYTES },
	{ "plain", shock_plain_frag_spv_spirv, SHOCK_PLAIN_FRAG_SPV_SPIRV_BYTES },
	{ "noloop", shock_noloop_frag_spv_spirv, SHOCK_NOLOOP_FRAG_SPV_SPIRV_BYTES },
	{ "tex", shock_tex_frag_spv_spirv, SHOCK_TEX_FRAG_SPV_SPIRV_BYTES },
	{ "flat", shock_flat_frag_spv_spirv, SHOCK_FLAT_FRAG_SPV_SPIRV_BYTES },
	{ "fullnoloop", shock_fullnoloop_frag_spv_spirv, SHOCK_FULLNOLOOP_FRAG_SPV_SPIRV_BYTES },
	{ "one", shock_one_frag_spv_spirv, SHOCK_ONE_FRAG_SPV_SPIRV_BYTES },
	{ "oneif", shock_oneif_frag_spv_spirv, SHOCK_ONEIF_FRAG_SPV_SPIRV_BYTES },
	{ "lean1", shock_lean1_frag_spv_spirv, SHOCK_LEAN1_FRAG_SPV_SPIRV_BYTES },
	{ "lean2", shock_lean2_frag_spv_spirv, SHOCK_LEAN2_FRAG_SPV_SPIRV_BYTES },
	{ "lean3", shock_lean3_frag_spv_spirv, SHOCK_LEAN3_FRAG_SPV_SPIRV_BYTES },
	{ "lean4", shock_lean4_frag_spv_spirv, SHOCK_LEAN4_FRAG_SPV_SPIRV_BYTES },
	{ "lean5", shock_lean5_frag_spv_spirv, SHOCK_LEAN5_FRAG_SPV_SPIRV_BYTES },
};
enum { VARIANTS = sizeof variants / sizeof variants[0], WORLD_PROGRAM = 1 };
static VkShaderModule variant_modules[VARIANTS];

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

static void log_gpu_state(const char *when)
{
	volatile const uint32_t *const registers = (volatile const uint32_t *)0x7FC80000u;
	xsdk_log_write("vkreplay: %s: RBBM_STATUS %08X, ring read 0x%X write 0x%X", when,
		(unsigned)registers[0x1740 / 4], (unsigned)registers[0x710 / 4], (unsigned)registers[0x714 / 4]);
}

static uint8_t *read_file(const char *path, uint32_t *size)
{
	FILE *const file = fopen(path, "rb");
	if (!file)
		return NULL;
	(void)fseek(file, 0, SEEK_END);
	const long length = ftell(file);
	(void)fseek(file, 0, SEEK_SET);
	uint8_t *const data = length > 0 ? malloc((size_t)length + 1) : NULL;
	if (data && fread(data, 1, (size_t)length, file) != (size_t)length) {
		free(data);
		fclose(file);
		return NULL;
	}
	fclose(file);
	if (data)
		data[length] = 0;
	*size = (uint32_t)length;
	return data;
}

// The dump's words are little-endian; its blobs are the console's own bytes.
typedef struct {
	const uint8_t *data;
	uint32_t size;
	uint32_t at;
	bool bad;
} Reader;

static uint32_t word(Reader *r)
{
	if (r->at + 4 > r->size) {
		r->bad = true;
		return 0;
	}
	const uint8_t *const p = r->data + r->at;
	r->at += 4;
	return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

static float real(Reader *r)
{
	const uint32_t bits = word(r);
	float value;
	memcpy(&value, &bits, 4);
	return value;
}

static const uint8_t *blob(Reader *r, uint32_t *bytes)
{
	*bytes = word(r);
	const uint32_t padded = (*bytes + 3) & ~3u;
	if (r->at + padded > r->size) {
		r->bad = true;
		return NULL;
	}
	const uint8_t *const p = r->data + r->at;
	r->at += padded;
	return p;
}

// SPIR-V words, swapped to the console's order into memory of their own.
static uint32_t *spirv(Reader *r, uint32_t *bytes)
{
	const uint32_t count = word(r);
	uint32_t *const words = malloc(count * 4 + 4);
	for (uint32_t i = 0; words && i < count; i++)
		words[i] = word(r);
	*bytes = count * 4;
	return words;
}

static bool parse(const Harness *harness, const uint8_t *data, uint32_t size)
{
	Reader r = { data, size, 0, false };
	while (r.at < size && !r.bad) {
		const uint32_t tag = word(&r);
		if (tag == TAG_FRAM) {
			replay.frame_size[0] = word(&r);
			replay.frame_size[1] = word(&r);
			const uint32_t color = word(&r), depth = word(&r), samples = word(&r);
			xsdk_log_write("vkreplay: frame %ux%u, formats %u/%u, %u samples", (unsigned)replay.frame_size[0],
				(unsigned)replay.frame_size[1], (unsigned)color, (unsigned)depth, (unsigned)samples);
		} else if (tag == TAG_CAPB || tag == TAG_CAPE) {
			xsdk_log_write("vkreplay: the dump has a depth capture, which vkreplay doesn't draw yet");
			return false;
		} else if (tag == TAG_CLER) {
			if (replay.clear_count == MAX_CLEARS)
				return false;
			Clear *const c = &replay.clears[replay.clear_count];
			c->color = word(&r);
			c->depth = word(&r);
			for (uint32_t i = 0; i < 4; i++)
				c->rgba[i] = real(&r);
			for (uint32_t i = 0; i < 4; i++)
				c->rect[i] = (int32_t)word(&r);
			replay.steps[replay.step_count++] = (Step){ true, replay.clear_count++ };
		} else if (tag == TAG_PROG) {
			const uint32_t id = word(&r);
			if (id >= MAX_PROGRAMS)
				return false;
			Program *const p = &replay.programs[id];
			uint32_t vertex_bytes = 0, fragment_bytes = 0;
			uint32_t *const vertex = spirv(&r, &vertex_bytes);
			uint32_t *const fragment = spirv(&r, &fragment_bytes);
			p->binding_count = word(&r);
			if (p->binding_count > MAX_BINDINGS)
				return false;
			for (uint32_t i = 0; i < p->binding_count; i++)
				for (uint32_t k = 0; k < 3; k++)
					p->bindings[i][k] = word(&r);
			bool ok = harness_ok("vertex module", harness_shader(harness, vertex, vertex_bytes, &p->vertex)) &&
				harness_ok("fragment module", harness_shader(harness, fragment, fragment_bytes, &p->fragment));
			free(vertex);
			free(fragment);
			if (!ok)
				return false;
			VkDescriptorSetLayoutBinding bindings[MAX_BINDINGS];
			for (uint32_t i = 0; i < p->binding_count; i++)
				bindings[i] = (VkDescriptorSetLayoutBinding){ .binding = p->bindings[i][0],
					.descriptorType = (VkDescriptorType)p->bindings[i][1],
					.descriptorCount = 1,
					.stageFlags = p->bindings[i][2] };
			const VkDescriptorSetLayoutCreateInfo set_info = { .sType =
										VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
				.bindingCount = p->binding_count,
				.pBindings = bindings };
			const VkPushConstantRange push = { VK_SHADER_STAGE_VERTEX_BIT, 0, 4 };
			ok = harness_ok("set layout", vkCreateDescriptorSetLayout(harness->device, &set_info, NULL, &p->set_layout));
			const VkPipelineLayoutCreateInfo layout_info = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
				.setLayoutCount = 1,
				.pSetLayouts = &p->set_layout,
				.pushConstantRangeCount = 1,
				.pPushConstantRanges = &push };
			ok = ok && harness_ok("pipeline layout", vkCreatePipelineLayout(harness->device, &layout_info, NULL, &p->layout));
			if (!ok)
				return false;
			p->used = true;
		} else if (tag == TAG_DRAW) {
			if (replay.draw_count == MAX_DRAWS)
				return false;
			Draw *const d = &replay.draws[replay.draw_count];
			d->program = word(&r);
			d->topology = word(&r);
			d->count = word(&r);
			d->stride = word(&r);
			d->attribute_count = word(&r);
			if (d->attribute_count > MAX_ATTRIBUTES || d->program >= MAX_PROGRAMS)
				return false;
			for (uint32_t i = 0; i < d->attribute_count; i++)
				for (uint32_t k = 0; k < 4; k++)
					d->attributes[i][k] = word(&r);
			d->cull[0] = word(&r);
			d->cull[1] = word(&r);
			d->depth_test = word(&r);
			d->depth_write = word(&r);
			d->depth_compare = word(&r);
			d->blend = word(&r);
			for (uint32_t i = 0; i < 4; i++)
				d->blend_factors[i] = word(&r);
			d->color_mask = word(&r);
			d->depth_bias = word(&r);
			d->bias[0] = real(&r);
			d->bias[1] = real(&r);
			for (uint32_t i = 0; i < 3; i++)
				d->formats[i] = word(&r);
			for (uint32_t i = 0; i < 6; i++)
				d->viewport[i] = real(&r);
			for (uint32_t i = 0; i < 4; i++)
				d->scissor[i] = (int32_t)word(&r);
			d->vertices = blob(&r, &d->vertex_bytes);
			d->indices = blob(&r, &d->index_bytes);
			d->blocks[0] = blob(&r, &d->block_bytes[0]);
			d->blocks[1] = blob(&r, &d->block_bytes[1]);
			d->sampler_count = word(&r);
			if (d->sampler_count > MAX_SAMPLERS)
				return false;
			for (uint32_t i = 0; i < d->sampler_count; i++)
				for (uint32_t k = 0; k < 7; k++)
					d->samplers[i][k] = word(&r);
			replay.steps[replay.step_count++] = (Step){ false, replay.draw_count++ };
		} else {
			xsdk_log_write("vkreplay: unknown record %08X at %u", (unsigned)tag, (unsigned)(r.at - 4));
			return false;
		}
	}
	return !r.bad;
}

// The image the dump's sampler names, made the first time: a gradient at its size, its levels and its layers.
static Image *image_for(const Harness *harness, uint32_t width, uint32_t height, uint32_t levels, uint32_t layers)
{
	for (uint32_t i = 0; i < replay.image_count; i++) {
		Image *const m = &replay.images[i];
		if (m->width == width && m->height == height && m->levels == levels && m->layers == layers)
			return m;
	}
	if (replay.image_count == MAX_IMAGES)
		return NULL;
	Image *const m = &replay.images[replay.image_count];
	*m = (Image){ .width = width, .height = height, .levels = levels, .layers = layers };
	const bool cube = layers == 6;
	const VkImageCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.flags = cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = VK_FORMAT_R8G8B8A8_UNORM,
		.extent = { width, height, 1 },
		.mipLevels = levels,
		.arrayLayers = layers,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
	};
	if (!harness_ok("image", vkCreateImage(harness->device, &info, NULL, &m->image)))
		return NULL;
	VkMemoryRequirements requirements;
	vkGetImageMemoryRequirements(harness->device, m->image, &requirements);
	const VkMemoryAllocateInfo allocate = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize = requirements.size };
	if (!harness_ok("image memory", vkAllocateMemory(harness->device, &allocate, NULL, &m->memory)) ||
		!harness_ok("bind image", vkBindImageMemory(harness->device, m->image, m->memory, 0)))
		return NULL;
	const VkImageViewCreateInfo view_info = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = m->image,
		.viewType = cube ? VK_IMAGE_VIEW_TYPE_CUBE : VK_IMAGE_VIEW_TYPE_2D,
		.format = VK_FORMAT_R8G8B8A8_UNORM,
		.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, levels, 0, layers },
	};
	if (!harness_ok("image view", vkCreateImageView(harness->device, &view_info, NULL, &m->view)))
		return NULL;

	// Every level and layer from one staging buffer.
	uint32_t total = 0;
	for (uint32_t level = 0; level < levels; level++) {
		const uint32_t w = width >> level ? width >> level : 1, h = height >> level ? height >> level : 1;
		total += w * h * 4 * layers;
	}
	uint8_t *const texels = malloc(total);
	if (!texels)
		return NULL;
	VkBufferImageCopy regions[16];
	uint32_t offset = 0;
	for (uint32_t level = 0; level < levels && level < 16; level++) {
		const uint32_t w = width >> level ? width >> level : 1, h = height >> level ? height >> level : 1;
		for (uint32_t y = 0; y < h * layers; y++)
			for (uint32_t x = 0; x < w; x++) {
				uint8_t *const t = texels + offset + (y * w + x) * 4;
				t[0] = (uint8_t)(64 + x * 191 / w), t[1] = (uint8_t)(64 + (y % h) * 191 / h), t[2] = 128, t[3] = 255;
			}
		regions[level] = (VkBufferImageCopy){ .bufferOffset = offset,
			.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, level, 0, layers },
			.imageExtent = { w, h, 1 } };
		offset += w * h * 4 * layers;
	}
	VkBuffer staging;
	VkDeviceMemory staging_memory;
	bool ok = harness_ok("staging",
		harness_buffer(harness, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, texels, total, &staging, &staging_memory));
	free(texels);
	if (!ok)
		return NULL;
	const VkCommandBufferAllocateInfo command_info = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = harness->pool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1 };
	VkCommandBuffer commands;
	if (!harness_ok("upload commands", vkAllocateCommandBuffers(harness->device, &command_info, &commands)))
		return NULL;
	const VkCommandBufferBeginInfo begin = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	(void)vkBeginCommandBuffer(commands, &begin);
	VkImageMemoryBarrier barrier = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
		.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = m->image,
		.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, levels, 0, layers },
	};
	vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0,
		NULL, 1, &barrier);
	vkCmdCopyBufferToImage(
		commands, staging, m->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, levels < 16 ? levels : 16, regions);
	barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, NULL,
		0, NULL, 1, &barrier);
	ok = harness_ok("upload recording", vkEndCommandBuffer(commands));
	const VkSubmitInfo submit = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.commandBufferCount = 1,
		.pCommandBuffers = &commands };
	ok = ok && harness_ok("upload submit", vkQueueSubmit(harness->queue, 1, &submit, harness->fence)) &&
		harness_ok("upload wait", vkWaitForFences(harness->device, 1, &harness->fence, VK_TRUE, 5000000000ull));
	(void)vkResetFences(harness->device, 1, &harness->fence);
	vkFreeCommandBuffers(harness->device, harness->pool, 1, &commands);
	vkDestroyBuffer(harness->device, staging, NULL);
	vkFreeMemory(harness->device, staging_memory, NULL);
	if (!ok)
		return NULL;
	replay.image_count++;
	return m;
}

// The engine's samplers: filter 0 nearest, 1 linear, 2 linear mipmapped (8x anisotropy where the device has it);
// wrap repeats, else clamps.
static VkSampler sampler_for(const Harness *harness, uint32_t filter, uint32_t wrap)
{
	VkSampler *const s = &replay.samplers[filter > 2 ? 2 : filter][wrap ? 1 : 0];
	if (*s)
		return *s;
	const VkSamplerAddressMode mode = wrap ? VK_SAMPLER_ADDRESS_MODE_REPEAT : VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	const VkSamplerCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.magFilter = filter ? VK_FILTER_LINEAR : VK_FILTER_NEAREST,
		.minFilter = filter ? VK_FILTER_LINEAR : VK_FILTER_NEAREST,
		.mipmapMode = filter == 2 ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST,
		.addressModeU = mode,
		.addressModeV = mode,
		.addressModeW = mode,
		.anisotropyEnable = filter == 2,
		.maxAnisotropy = filter == 2 ? 8.0f : 1.0f,
		.maxLod = filter == 2 ? 1000.0f : 0.0f,
	};
	(void)harness_ok("sampler", vkCreateSampler(harness->device, &info, NULL, s));
	return *s;
}

static bool build(const Harness *harness)
{
	bool ok = harness_ok("depth image",
		harness_depth_image(harness, &replay.depth, &replay.depth_memory, &replay.depth_view));
	ok = ok &&
		harness_ok("render pass",
			harness_depth_render_pass(harness, replay.depth_view, &replay.render_pass, replay.framebuffers));
	static const uint8_t zeros[16] = { 0 };
	ok = ok &&
		harness_ok("zero buffer",
			harness_buffer(harness, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, zeros, sizeof zeros, &replay.zero,
				&replay.zero_memory));

	// Every draw's blocks in one buffer, each on the device's alignment.
	uint32_t uniform_bytes = 0;
	for (uint32_t i = 0; i < replay.draw_count; i++)
		for (uint32_t b = 0; b < 2; b++) {
			replay.draws[i].block_offsets[b] = uniform_bytes;
			uniform_bytes += (replay.draws[i].block_bytes[b] + UNIFORM_ALIGN - 1) & ~(uint32_t)(UNIFORM_ALIGN - 1);
		}
	uint8_t *const blocks = calloc(1, uniform_bytes + UNIFORM_ALIGN);
	ok = ok && blocks;
	for (uint32_t i = 0; ok && i < replay.draw_count; i++)
		for (uint32_t b = 0; b < 2; b++)
			if (replay.draws[i].block_bytes[b])
				memcpy(blocks + replay.draws[i].block_offsets[b], replay.draws[i].blocks[b],
					replay.draws[i].block_bytes[b]);
	ok = ok &&
		harness_ok("uniform buffer",
			harness_buffer(harness, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, blocks, uniform_bytes + UNIFORM_ALIGN,
				&replay.uniforms, &replay.uniform_memory));
	free(blocks);

	const VkDescriptorPoolSize sizes[] = {
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2 * MAX_DRAWS },
		{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, MAX_SAMPLERS * MAX_DRAWS },
	};
	const VkDescriptorPoolCreateInfo pool_info = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = MAX_DRAWS,
		.poolSizeCount = 2,
		.pPoolSizes = sizes };
	ok = ok && harness_ok("descriptor pool", vkCreateDescriptorPool(harness->device, &pool_info, NULL, &replay.pool));

	for (uint32_t i = 0; ok && i < replay.draw_count; i++) {
		Draw *const d = &replay.draws[i];
		const Program *const p = &replay.programs[d->program];
		if (!p->used) {
			xsdk_log_write("vkreplay: draw %u's program %u isn't in the dump", (unsigned)i, (unsigned)d->program);
			return false;
		}
		ok = harness_ok("vertex buffer",
			harness_buffer(harness, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, d->vertices, d->vertex_bytes ? d->vertex_bytes : 4,
				&d->vertex_buffer, &d->vertex_memory));
		// Positions are the first three floats of a vertex, in the console's byte order as the dump keeps them.
		if (ok && !d->index_bytes && d->topology == 0 && d->stride >= 12) {
			uint8_t *const parts[2] = { malloc(d->vertex_bytes + 4), malloc(d->vertex_bytes + 4) };
			ok = parts[0] && parts[1];
			for (uint32_t t = 0; ok && t + 1 < d->count / 3 + 1 && (t + 1) * 3 <= d->count; t++) {
				float p[3][3];
				for (uint32_t k = 0; k < 3; k++)
					memcpy(p[k], d->vertices + (t * 3 + k) * d->stride, 12);
				const float e1[3] = { p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2] };
				const float e2[3] = { p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2] };
				const float cross[3] = { e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2],
					e1[0] * e2[1] - e1[1] * e2[0] };
				const float size = (cross[0] < 0 ? -cross[0] : cross[0]) + (cross[1] < 0 ? -cross[1] : cross[1]) +
					(cross[2] < 0 ? -cross[2] : cross[2]);
				const uint32_t part = size < 1e-6f;
				memcpy(parts[part] + d->split_counts[part] * d->stride, d->vertices + t * 3 * d->stride, 3 * d->stride);
				d->split_counts[part] += 3;
			}
			for (uint32_t part = 0; ok && part < 2; part++)
				ok = harness_ok("split buffer",
					harness_buffer(harness, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, parts[part],
						d->split_counts[part] ? d->split_counts[part] * d->stride : 4, &d->split_buffers[part],
						&d->split_memory[part]));
			free(parts[0]);
			free(parts[1]);
		}
		if (ok && d->index_bytes)
			ok = harness_ok("index buffer",
				harness_buffer(harness, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, d->indices, d->index_bytes, &d->index_buffer,
					&d->index_memory));
		const VkDescriptorSetAllocateInfo allocate = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorPool = replay.pool,
			.descriptorSetCount = 1,
			.pSetLayouts = &p->set_layout };
		ok = ok && harness_ok("descriptor set", vkAllocateDescriptorSets(harness->device, &allocate, &d->set));
		VkDescriptorBufferInfo buffers[2];
		VkDescriptorImageInfo images[MAX_SAMPLERS];
		VkWriteDescriptorSet writes[2 + MAX_SAMPLERS];
		uint32_t write_count = 0;
		for (uint32_t b = 0; ok && b < 2; b++) {
			if (!d->block_bytes[b])
				continue;
			buffers[b] = (VkDescriptorBufferInfo){ replay.uniforms, d->block_offsets[b], d->block_bytes[b] };
			writes[write_count++] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = d->set,
				.dstBinding = b,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.pBufferInfo = &buffers[b] };
		}
		for (uint32_t s = 0; ok && s < d->sampler_count; s++) {
			const uint32_t *const t = d->samplers[s];
			const Image *const m = image_for(harness, t[1], t[2], t[3], t[4]);
			ok = m != NULL;
			if (!ok)
				break;
			images[s] = (VkDescriptorImageInfo){ sampler_for(harness, t[5], t[6]), m->view,
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
			writes[write_count++] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = d->set,
				.dstBinding = t[0],
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.pImageInfo = &images[s] };
		}
		if (ok)
			vkUpdateDescriptorSets(harness->device, write_count, writes, 0, NULL);
	}
	return ok;
}

static VkResult create_pipeline(const Harness *harness, const Draw *d, const Phase *phase, VkPipeline *pipeline)
{
	const Program *const p = &replay.programs[d->program];
	VkVertexInputBindingDescription bindings[2] = {
		{ 0, d->stride, VK_VERTEX_INPUT_RATE_VERTEX },
		{ 1, 0, VK_VERTEX_INPUT_RATE_VERTEX },
	};
	uint32_t binding_count = 1;
	VkVertexInputAttributeDescription attributes[MAX_ATTRIBUTES];
	for (uint32_t i = 0; i < d->attribute_count; i++) {
		attributes[i] = (VkVertexInputAttributeDescription){ .location = d->attributes[i][0],
			.binding = d->attributes[i][1],
			.format = (VkFormat)d->attributes[i][2],
			.offset = d->attributes[i][3] };
		if (d->attributes[i][1] == 1)
			binding_count = 2;
	}
	const VkPipelineInputAssemblyStateCreateInfo assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = d->topology ? VK_PRIMITIVE_TOPOLOGY_LINE_LIST : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
	};
	const VkPipelineRasterizationStateCreateInfo raster = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = phase->cull ? VK_CULL_MODE_BACK_BIT : d->cull[0],
		.frontFace = (VkFrontFace)d->cull[1],
		.depthBiasEnable = d->depth_bias,
		.lineWidth = 1.0f,
	};
	const VkPipelineDepthStencilStateCreateInfo depth = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = d->depth_test && !phase->no_depth_test,
		.depthWriteEnable = d->depth_write && !phase->no_depth_write,
		.depthCompareOp = (VkCompareOp)d->depth_compare,
	};
	const VkPipelineColorBlendAttachmentState blend = {
		.blendEnable = d->blend && !phase->no_blend,
		.srcColorBlendFactor = (VkBlendFactor)d->blend_factors[0],
		.dstColorBlendFactor = (VkBlendFactor)d->blend_factors[1],
		.colorBlendOp = VK_BLEND_OP_ADD,
		.srcAlphaBlendFactor = (VkBlendFactor)d->blend_factors[2],
		.dstAlphaBlendFactor = (VkBlendFactor)d->blend_factors[3],
		.alphaBlendOp = VK_BLEND_OP_ADD,
		.colorWriteMask = phase->no_color ? 0 : d->color_mask,
	};
	static const VkDynamicState dynamic[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
		VK_DYNAMIC_STATE_DEPTH_BIAS };
	// Made here, not by harness_pipeline, which makes its own modules: the dump's are made already.
	const VkPipelineShaderStageCreateInfo stages[2] = {
		{ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = p->vertex,
			.pName = "main" },
		{ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = phase->shader && d->program == WORLD_PROGRAM ? variant_modules[phase->shader] : p->fragment,
			.pName = "main" },
	};
	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = binding_count,
		.pVertexBindingDescriptions = bindings,
		.vertexAttributeDescriptionCount = d->attribute_count,
		.pVertexAttributeDescriptions = attributes,
	};
	const VkPipelineViewportStateCreateInfo viewport = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};
	const VkPipelineMultisampleStateCreateInfo multisample = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = d->formats[2] ? (VkSampleCountFlagBits)d->formats[2] : VK_SAMPLE_COUNT_1_BIT,
	};
	const VkPipelineColorBlendStateCreateInfo blend_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &blend,
	};
	const VkPipelineDynamicStateCreateInfo dynamic_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = 3,
		.pDynamicStates = dynamic,
	};
	const VkGraphicsPipelineCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = 2,
		.pStages = stages,
		.pVertexInputState = &vertex_input,
		.pInputAssemblyState = &assembly,
		.pViewportState = &viewport,
		.pRasterizationState = &raster,
		.pMultisampleState = &multisample,
		.pDepthStencilState = &depth,
		.pColorBlendState = &blend_state,
		.pDynamicState = &dynamic_state,
		.layout = p->layout,
		.renderPass = replay.render_pass,
	};
	return vkCreateGraphicsPipelines(harness->device, VK_NULL_HANDLE, 1, &info, NULL, pipeline);
}

static bool in_phase(const Phase *phase, uint32_t index)
{
	const Draw *const d = &replay.draws[index];
	return (phase->program < 0 || (uint32_t)phase->program == d->program) && index >= phase->first &&
		index <= phase->last;
}

static bool record(const Harness *harness, const Phase *phase, VkPipeline *pipelines, VkCommandBuffer *commands)
{
	for (uint32_t i = 0; i < replay.draw_count; i++) {
		pipelines[i] = VK_NULL_HANDLE;
		if (in_phase(phase, i) && !harness_ok("pipeline", create_pipeline(harness, &replay.draws[i], phase, &pipelines[i])))
			return false;
	}
	if (!harness_ok("command buffers", harness_command_buffers(harness, commands)))
		return false;
	static const VkClearValue clears[2] = {
		{ .color = { .float32 = { 0.0f, 0.0f, 0.0f, 1.0f } } },
		{ .depthStencil = { 1.0f, 0 } },
	};
	for (uint32_t image = 0; image < harness->image_count; image++) {
		const VkCommandBuffer c = commands[image];
		const VkRenderPassBeginInfo pass = {
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = replay.render_pass,
			.framebuffer = replay.framebuffers[image],
			.renderArea = { { 0, 0 }, { HARNESS_WIDTH, HARNESS_HEIGHT } },
			.clearValueCount = 2,
			.pClearValues = clears,
		};
		vkCmdBeginRenderPass(c, &pass, VK_SUBPASS_CONTENTS_INLINE);
		for (uint32_t s = 0; s < replay.step_count; s++) {
			const Step step = replay.steps[s];
			if (step.is_clear) {
				const Clear *const k = &replay.clears[step.index];
				if (phase->no_clear)
					continue;
				VkClearAttachment attachments[2];
				uint32_t count = 0;
				if (k->color)
					attachments[count++] = (VkClearAttachment){ VK_IMAGE_ASPECT_COLOR_BIT, 0,
						{ .color = { .float32 = { k->rgba[0], k->rgba[1], k->rgba[2], k->rgba[3] } } } };
				if (k->depth)
					attachments[count++] =
						(VkClearAttachment){ VK_IMAGE_ASPECT_DEPTH_BIT, 0, { .depthStencil = { 1.0f, 0 } } };
				const VkClearRect rect = { { { k->rect[0], k->rect[1] }, { (uint32_t)k->rect[2], (uint32_t)k->rect[3] } },
					0, 1 };
				if (count)
					vkCmdClearAttachments(c, count, attachments, 1, &rect);
				continue;
			}
			if (!pipelines[step.index])
				continue;
			const Draw *const d = &replay.draws[step.index];
			const VkViewport viewport = { d->viewport[0], d->viewport[1], d->viewport[2], d->viewport[3], d->viewport[4],
				d->viewport[5] };
			const VkRect2D scissor = { { d->scissor[0], d->scissor[1] }, { (uint32_t)d->scissor[2], (uint32_t)d->scissor[3] } };
			vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines[step.index]);
			vkCmdSetViewport(c, 0, 1, &viewport);
			vkCmdSetScissor(c, 0, 1, &scissor);
			vkCmdSetDepthBias(c, d->bias[0], 0.0f, d->bias[1]);
			vkCmdBindDescriptorSets(
				c, VK_PIPELINE_BIND_POINT_GRAPHICS, replay.programs[d->program].layout, 0, 1, &d->set, 0, NULL);
			static const float flip_y = -1.0f;
			vkCmdPushConstants(c, replay.programs[d->program].layout, VK_SHADER_STAGE_VERTEX_BIT, 0, 4, &flip_y);
			const bool split = phase->split >= 0 && d->split_buffers[0];
			const VkBuffer buffers[2] = { split ? d->split_buffers[phase->split] : d->vertex_buffer, replay.zero };
			const VkDeviceSize offsets[2] = { 0, 0 };
			vkCmdBindVertexBuffers(c, 0, 2, buffers, offsets);
			if (d->index_bytes) {
				vkCmdBindIndexBuffer(c, d->index_buffer, 0, VK_INDEX_TYPE_UINT32);
				vkCmdDrawIndexed(c, d->count, 1, 0, 0, 0);
			} else {
				const uint32_t total = split ? d->split_counts[phase->split] : d->count;
				const uint32_t from = phase->vertex_first / 3 * 3;
				const uint32_t to = phase->vertex_last < total - 1 ? phase->vertex_last : total - 1;
				const uint32_t count = to >= from ? (to - from + 1) / 3 * 3 : 0;
				if (count)
					vkCmdDraw(c, count, 1, from, 0);
			}
		}
		vkCmdEndRenderPass(c);
		if (!harness_ok("recording", vkEndCommandBuffer(c)))
			return false;
	}
	return true;
}

static uint32_t frame(const Harness *harness, const VkCommandBuffer *commands)
{
	uint32_t index = 0;
	const uint64_t timeout_ns = (uint64_t)FRAME_TIMEOUT_MS * 1000000u;
	if (!harness_ok("vkAcquireNextImageKHR",
			vkAcquireNextImageKHR(harness->device, harness->swapchain, timeout_ns, VK_NULL_HANDLE, VK_NULL_HANDLE, &index)))
		return 0;
	const VkSubmitInfo submit = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.commandBufferCount = 1,
		.pCommandBuffers = &commands[index] };
	const uint64_t start = now_us();
	if (!harness_ok("vkQueueSubmit", vkQueueSubmit(harness->queue, 1, &submit, harness->fence)))
		return 0;
	if (!harness_ok("vkWaitForFences", vkWaitForFences(harness->device, 1, &harness->fence, VK_TRUE, timeout_ns))) {
		log_gpu_state("the frame didn't finish");
		return 0;
	}
	const uint64_t took = now_us() - start;
	(void)vkResetFences(harness->device, 1, &harness->fence);
	const VkPresentInfoKHR present = { .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
		.swapchainCount = 1,
		.pSwapchains = &harness->swapchain,
		.pImageIndices = &index };
	if (!harness_ok("vkQueuePresentKHR", vkQueuePresentKHR(harness->queue, &present)))
		return 0;
	return took ? (uint32_t)took : 1;
}

// The world program's spotlightCount, at this byte of its fragment block (the dump's SPIR-V's member offset).
enum { SPOTLIGHT_COUNT_OFFSET = 992 };

static void set_spot_count(const Harness *harness, int32_t count)
{
	uint8_t *mapped = NULL;
	if (!harness_ok("map uniforms",
			vkMapMemory(harness->device, replay.uniform_memory, 0, VK_WHOLE_SIZE, 0, (void **)&mapped)))
		return;
	for (uint32_t i = 0; i < replay.draw_count; i++) {
		const Draw *const d = &replay.draws[i];
		if (d->program != WORLD_PROGRAM || d->block_bytes[1] < SPOTLIGHT_COUNT_OFFSET + 4)
			continue;
		float value;
		memcpy(&value, d->blocks[1] + SPOTLIGHT_COUNT_OFFSET, 4);
		if (count >= 0)
			value = (float)count;
		memcpy(mapped + d->block_offsets[1] + SPOTLIGHT_COUNT_OFFSET, &value, 4);
	}
	vkUnmapMemory(harness->device, replay.uniform_memory);
}

static bool run(const Harness *harness, const Phase *phase)
{
	xsdk_log_write("vkreplay: phase \"%s\"", phase->name);
	set_spot_count(harness, phase->spot_count);
	static VkPipeline pipelines[MAX_DRAWS];
	VkCommandBuffer commands[HARNESS_MAX_IMAGES] = { VK_NULL_HANDLE };
	if (!record(harness, phase, pipelines, commands))
		return false;
	uint32_t total = 0;
	bool whole = true;
	for (uint32_t step = 0; step < STEPS && whole; step++) {
		uint64_t step_us = 0;
		for (uint32_t i = 0; i < FRAMES_A_STEP && whole; i++) {
			const uint32_t took = frame(harness, commands);
			whole = took != 0;
			if (total < TIMED_FRAMES)
				xsdk_log_write("vkreplay: \"%s\": frame %u took %u us", phase->name, (unsigned)total, (unsigned)took);
			total += whole;
			step_us += took;
		}
		xsdk_log_write("vkreplay: \"%s\": %u frames, %u us a frame", phase->name, (unsigned)total,
			(unsigned)(step_us / FRAMES_A_STEP));
	}
	check(phase->name, whole);
	if (whole) {
		vkFreeCommandBuffers(harness->device, harness->pool, harness->image_count, commands);
		for (uint32_t i = 0; i < replay.draw_count; i++)
			if (pipelines[i])
				vkDestroyPipeline(harness->device, pipelines[i], NULL);
	}
	return whole;
}

static uint32_t read_phases(Phase *phases)
{
	uint32_t size = 0;
	char *const text = (char *)read_file("game:\\replay.txt", &size);
	uint32_t count = 0;
	char *line = text;
	while (line && *line && count < MAX_PHASES) {
		char *const end = strchr(line, '\n');
		if (end)
			*end = 0;
		Phase *const p = &phases[count];
		*p = (Phase){ .program = -1, .first = 0, .last = 0xFFFFFFFFu, .vertex_last = 0xFFFFFFFFu, .split = -1, .spot_count = -1 };
		snprintf(p->name, sizeof p->name, "%s", line);
		for (char *c = p->name; *c; c++)
			if (*c == '\r')
				*c = 0;
		char words[96];
		snprintf(words, sizeof words, "%s", p->name);
		for (char *w = strtok(words, " \t"); w; w = strtok(NULL, " \t")) {
			unsigned a = 0, b = 0;
			if (!strcmp(w, "noblend"))
				p->no_blend = true;
			else if (!strcmp(w, "nodepthwrite"))
				p->no_depth_write = true;
			else if (!strcmp(w, "nodepthtest"))
				p->no_depth_test = true;
			else if (!strcmp(w, "cull"))
				p->cull = true;
			else if (!strcmp(w, "mask=0"))
				p->no_color = true;
			else if (!strcmp(w, "noclear"))
				p->no_clear = true;
			else if (!strncmp(w, "shader=", 7)) {
				for (uint32_t v = 1; v < VARIANTS; v++)
					if (!strcmp(w + 7, variants[v].name))
						p->shader = v;
			} else if (sscanf(w, "spotcount=%u", &a) == 1)
				p->spot_count = (int32_t)a;
			else if (!strcmp(w, "nodegen"))
				p->split = 0;
			else if (!strcmp(w, "onlydegen"))
				p->split = 1;
			else if (sscanf(w, "program=%u", &a) == 1)
				p->program = (int32_t)a;
			else if (sscanf(w, "draws=%u-%u", &a, &b) == 2)
				p->first = a, p->last = b;
			else if (sscanf(w, "verts=%u-%u", &a, &b) == 2)
				p->vertex_first = a, p->vertex_last = b;
		}
		if (!p->name[0])
			snprintf(p->name, sizeof p->name, "asis");
		count++;
		line = end ? end + 1 : NULL;
	}
	free(text);
	if (!count) {
		phases[0] = (Phase){ .program = -1, .first = 0, .last = 0xFFFFFFFFu, .vertex_last = 0xFFFFFFFFu, .split = -1, .spot_count = -1 };
		snprintf(phases[0].name, sizeof phases[0].name, "asis");
		count = 1;
	}
	return count;
}

int main(void)
{
	xsdk_log_write("vkreplay: shock2quest's dumped frame, drawn again");
	Harness harness = { .instance = VK_NULL_HANDLE };
	uint32_t size = 0;
	uint8_t *const data = read_file("game:\\draws.bin", &size);
	bool ok = data != NULL;
	check("vkreplay: draws.bin read", ok);
	ok = ok && harness_ok("device", harness_create_device(&harness)) &&
		harness_ok("swapchain", harness_create_swapchain(&harness, 2));
	ok = ok && parse(&harness, data, size);
	check("vkreplay: draws.bin parsed", ok);
	if (ok)
		xsdk_log_write("vkreplay: %u draws, %u clears", (unsigned)replay.draw_count, (unsigned)replay.clear_count);
	ok = ok && build(&harness);
	for (uint32_t v = 1; ok && v < VARIANTS; v++)
		ok = harness_ok("variant module", harness_shader(&harness, variants[v].spirv, variants[v].bytes, &variant_modules[v]));
	check("vkreplay: buffers, textures and sets made", ok);
	static Phase phases[MAX_PHASES];
	const uint32_t phase_count = ok ? read_phases(phases) : 0;
	for (uint32_t i = 0; ok && i < phase_count; i++)
		ok = run(&harness, &phases[i]);
	xsdk_log_write(ok ? "vkreplay: every phase ran" : "vkreplay: stopped");
	return ok ? 0 : 1;
}
