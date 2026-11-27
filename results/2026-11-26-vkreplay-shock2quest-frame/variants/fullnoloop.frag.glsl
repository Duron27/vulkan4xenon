
uniform bool pointShadowEnabled;
uniform highp sampler2D pointShadowAtlas;
// xyz position, w far plane (0 = slot unused).
uniform highp vec4 pointShadowLights[4];
uniform vec4 pointShadowColor[4];
uniform int lightShadowSlot[6];
uniform float pointShadowStrength;
const highp float POINT_SHADOW_NEAR = 0.5;
const int POINT_SHADOW_FACE = 512;
const int POINT_SHADOW_MAX = 4;
// The atlas: six faces across, a row per set and light.
const highp vec2 POINT_SHADOW_ATLAS = vec2(6 * POINT_SHADOW_FACE, 2 * POINT_SHADOW_MAX * POINT_SHADOW_FACE);

// Mirrors engine::scene::point_shadow::FACES. One return per function here:
// the Xbox 360's shader compiler inlines only single-exit functions.
// The face index is a float: the Xenos ALU has no integer select.
highp float pointShadowFace(highp vec3 d, out highp vec3 forward, out highp vec3 up) {
    highp vec3 a = abs(d);
    // Everything starts at face 5's values: xost doesn't follow a value set
    // only in the arms of nested ifs.
    highp float face = 5.0;
    forward = vec3(0.0, 0.0, -1.0);
    up = vec3(0.0, 1.0, 0.0);
    if (a.x >= a.y && a.x >= a.z) {
        face = d.x >= 0.0 ? 0.0 : 1.0;
        forward = vec3(d.x >= 0.0 ? 1.0 : -1.0, 0.0, 0.0);
    } else if (a.y >= a.z) {
        face = d.y >= 0.0 ? 2.0 : 3.0;
        forward = vec3(0.0, d.y >= 0.0 ? 1.0 : -1.0, 0.0);
        up = vec3(0.0, 0.0, d.y >= 0.0 ? 1.0 : -1.0);
    } else if (d.z >= 0.0) {
        face = 4.0;
        forward = vec3(0.0, 0.0, 1.0);
    }
    return face;
}

// 1 lit .. 0 shadowed, from `light`'s cube in `set` (0 every caster,
// 1 objects only).
highp float pointShadowVisibility(int light, int set, highp vec3 worldPos, vec3 normal) {
    highp float visibility = 1.0;
    // No shadow captures on the Xbox 360 (vk_backend::begin_depth_capture),
    // and both arms of a branch run there: leave the sampling out.
#ifndef SHOCK_XBOX360
    if (pointShadowEnabled && light >= 0) {
        highp vec4 l = pointShadowLights[light];
        highp float far = l.w;
        highp vec3 d = worldPos - l.xyz;
        highp float dist = length(d);
        if (far > 0.0 && dist < far) {
            // Normal offset by about a texel's footprint, toward the lamp's side:
            // keeps a lit surface from shadowing itself (acne).
            vec3 n = dot(normal, d) > 0.0 ? -normal : normal;
            d += n * (3.0 * dist / float(POINT_SHADOW_FACE));
            highp vec3 forward;
            highp vec3 up;
            highp float face = pointShadowFace(d, forward, up);
            highp float m = dot(d, forward);
            if (m > POINT_SHADOW_NEAR) {
                highp vec3 right = cross(forward, up);
                highp vec2 uv = vec2(dot(d, right), dot(d, up)) / m * 0.5 + 0.5;
                highp float size = float(POINT_SHADOW_FACE);
                highp vec2 tile = vec2(face, float(set * POINT_SHADOW_MAX + light)) * size;
                highp vec2 texel = tile + vec2(uv.x, 1.0 - uv.y) * size;
                highp float n2 = POINT_SHADOW_NEAR;
                highp float bias = 0.03 + 0.004 * m;
                highp float lit = 0.0;
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        highp vec2 t = clamp(texel + vec2(float(dx), float(dy)), tile + 0.5, tile + size - 0.5);
#ifdef SHOCK_XBOX360
                        // xost has no texelFetch: the same texel, sampled at its centre.
                        highp float z = textureLod(pointShadowAtlas, (floor(t) + 0.5) / POINT_SHADOW_ATLAS, 0.0).r * 2.0 - 1.0;
#else
                        highp float z = texelFetch(pointShadowAtlas, ivec2(t), 0).r * 2.0 - 1.0;
#endif
                        // The occluder's distance along the face axis.
                        highp float occluder = 2.0 * far * n2 / ((far + n2) - z * (far - n2));
                        lit += m - bias > occluder ? 0.0 : 1.0;
                    }
                }
                visibility = lit / 9.0;
            }
        }
    }
#endif
    return visibility;
}


uniform highp vec4 fogEyeDensity;
// One return: the Xbox 360's shader compiler inlines only single-exit functions.
highp float fogFactor(highp vec3 worldPos) {
    highp float fog = 1.0;
    if (fogEyeDensity.w > 0.0) {
        highp float d = distance(worldPos, fogEyeDensity.xyz);
        fog = max(exp(-fogEyeDensity.w * d), 0.12);
    }
    return fog;
}


uniform float bumpStrength;
float bumpHeight(sampler2D tex, vec2 uv) {
    return dot(texture(tex, uv).rgb, vec3(0.299, 0.587, 0.114));
}
// `depth` is how deep, in world units, full black sits below full white.
// One return: the Xbox 360's shader compiler inlines only single-exit functions.
vec3 bumpNormal(sampler2D tex, vec2 uv, highp vec3 pos, vec3 n, float depth) {
    vec3 bumped = n;
    if (bumpStrength > 0.0) {
        vec2 duvdx = dFdx(uv);
        vec2 duvdy = dFdy(uv);
        highp vec3 dpdx = dFdx(pos);
        highp vec3 dpdy = dFdy(pos);
        // Height change across one pixel, sampled per pixel (not per 2x2 quad,
        // which would show as blocks).
        float h = bumpHeight(tex, uv);
        float dhdx = (bumpHeight(tex, uv + duvdx) - h) * depth * bumpStrength;
        float dhdy = (bumpHeight(tex, uv + duvdy) - h) * depth * bumpStrength;
        highp vec3 r1 = cross(dpdy, n);
        highp vec3 r2 = cross(n, dpdx);
        highp float det = dot(dpdx, r1);
        if (abs(det) >= 1e-12) {
            highp vec3 grad = sign(det) * (dhdx * r1 + dhdy * r2);
            bumped = normalize(abs(det) * n - grad);
        }
    }
    return bumped;
}


vec3 environmentDirection(vec3 engineDirection) {
    return vec3(-engineDirection.z, engineDirection.y, -engineDirection.x);
}


uniform bool shineEnabled;
uniform sampler2D shineMask;
uniform sampler2D shineRamp;
uniform vec3 shineEye;
uniform vec3 shineTint;
uniform bool shineUnlit;
uniform bool shineAlphaBlend;
uniform float shinePasses;
uniform float shineSpecular;
uniform samplerCube shineEnvironment;
uniform float shineReflection;
uniform sampler2D shineVeins;
uniform float shineVeinStrength;
uniform float shineVeinSharpness;
uniform float shineVeinScale;

// Highlight exponent. Higher reads wetter but sparkles on low-poly meshes in
// a headset.
const float SHINE_POWER = 24.0;
// Gloss away from the authored glints, relative to 1 on them.
const float SHINE_BASE_GLOSS = 0.5;
// Mip the capture is reflected at: blurry, as a curved wet surface would
// show it, and forgiving of the capture having been taken elsewhere on the deck.
const float REFLECTION_LOD = 3.0;
// Schlick reflectance head-on (above water's 0.02 so it reads at a glance);
// grazing angles approach 1.
const float REFLECTION_F0 = 0.05;
// Vein mask response: the field between veins keeps SHINE_VEIN_BASE of the
// even highlight; a ridge peaks at BASE + GAIN.
const float SHINE_VEIN_BASE = 0.3;
const float SHINE_VEIN_GAIN = 1.8;
// Share of the vein response the environment reflection follows: half, so
// the surroundings still read as a mirror image rather than only on veins.
const float SHINE_VEIN_REFLECTION = 0.5;

bool shineHighlights() {
    return shineEnabled && shineSpecular > 0.0;
}

// Toksvig: interpolated vertex normals shorten where the surface bends
// within a triangle, so widen the lobe there instead of letting it flicker.
float shinePower(float normalLength) {
    return SHINE_POWER * normalLength / (normalLength + SHINE_POWER * (1.0 - normalLength));
}

// Blinn-Phong lobe of one light; `radiance` is its colour after attenuation.
// The energy-conserving scale keeps a widened lobe from brightening.
vec3 shineHighlight(vec3 radiance, vec3 lightDir, vec3 normal, vec3 position, float power) {
    // The normal nudge keeps a light straight behind the surface from
    // normalizing a zero vector.
    vec3 halfway = normalize(lightDir + normalize(shineEye - position) + 1e-4 * normal);
    // Fade in past the terminator rather than cutting off at it.
    float lit = clamp(dot(normal, lightDir) * 4.0, 0.0, 1.0);
    return radiance * pow(max(dot(normal, halfway), 0.0), power) * lit
        * (1.0 + power) / (1.0 + SHINE_POWER);
}

// `specular` is the summed highlight of every light.
// One return: the Xbox 360's shader compiler inlines only single-exit functions.
vec4 applyShine(vec4 base, vec3 light, vec3 specular, float opacity, vec2 uv, vec3 position, vec3 normal) {
    vec4 shaded = base;
    if (shineEnabled) {
        vec4 mask = texture(shineMask, uv);
        // A separate pass would alpha-test its own bitmap.
        if (!(mask.a < 0.1 && !shineHighlights() && shineReflection <= 0.0)) {
            vec3 toEye = normalize(shineEye - position);
            float facing = clamp(abs(dot(normal, toEye)), 0.0, 1.0);
            float alpha = mask.a < 0.1 ? 0.0 : mask.a * opacity * texture(shineRamp, vec2(facing, 0.5)).r;
            vec3 shine = min((shineUnlit ? mask.rgb : mask.rgb * light) * shineTint, vec3(1.0));
            // The whole surface is wet, in the mask's colour; its sparse alpha marks
            // the glossiest glints. The ramp only shapes the authored sheen.
            float gloss = mix(SHINE_BASE_GLOSS, 1.0, mask.a) * opacity;
            vec3 highlight = mask.rgb * gloss * specular * shineSpecular;
            // Uniform per draw: surfaces without veins skip the extra sample.
            vec3 veins = vec3(1.0);
            if (shineVeinStrength > 0.0) {
                float vein = texture(shineVeins, uv * shineVeinScale).r;
                // Sharpening thresholds the soft ridge into a thin, hard-edged line.
                float edge = mix(0.35, 0.04, shineVeinSharpness);
                vein = mix(vein, smoothstep(0.55 - edge, 0.55 + edge, vein), shineVeinSharpness);
                // Tint the ridge toward the surface's own hue so it reads wet, not white.
                vec3 hue = mask.rgb / max(max(mask.r, max(mask.g, mask.b)), 1e-3);
                vec3 ridge = (SHINE_VEIN_BASE + vein * SHINE_VEIN_GAIN) * mix(vec3(1.0), hue, 0.35);
                veins = mix(vec3(1.0), ridge, min(shineVeinStrength, 1.0));
                highlight *= veins;
            }
            if (shineReflection > 0.0) {
                vec3 mirrored = reflect(-toEye, normal);
                vec3 seen = textureLod(shineEnvironment, environmentDirection(mirrored), REFLECTION_LOD).rgb;
                float fresnel = mix(REFLECTION_F0, 1.0, pow(1.0 - facing, 5.0));
                // The capture is a lit photograph of the deck; the light that actually
                // reaches this surface keeps a dark room from reflecting a bright one.
                highlight += seen * mask.rgb * min(light, vec3(1.0)) * fresnel * gloss * shineReflection
                    * mix(vec3(1.0), veins, SHINE_VEIN_REFLECTION);
            }
            // Each stacked pass blended onto a clamped framebuffer.
            vec3 color = min(base.rgb, vec3(1.0));
            if (shineAlphaBlend) {
                float covered = 1.0 - pow(1.0 - alpha, shinePasses);
                float outAlpha = base.a * (1.0 - covered) + covered;
                vec3 blended = color * base.a * (1.0 - covered) + shine * covered + highlight;
                shaded = vec4(blended / max(outAlpha, 1.0 / 255.0), outAlpha);
            } else {
                float coverage = max(base.a, 1.0 / 255.0);
                // The blend scales colour by the base's coverage; an additive pass does
                // not. Exact until a faded base's pre-divided colour saturates.
                shaded = vec4(color + (shinePasses * shine * alpha + highlight) / coverage, base.a);
            }
        }
    }
    return shaded;
}

// Analytic projection profile shared by walls, props and skinned meshes.
// Keep the whole cone calculation highp: mediump cosines near 1 quantize a
// narrow beam into rings on GLES, even when the world positions are highp.
// One return: the Xbox 360's shader compiler inlines only single-exit functions.
highp float flashlightCone(highp float cosine, highp float innerCosine, highp float outerCosine) {
    highp float cone = 0.0;
    if (cosine > 0.0) {
        // Squared radius on a plane one unit in front of the lamp. A radial
        // profile needs no projection basis, texture fetch or resolution limit.
        highp float radius2 = max(1.0 - cosine * cosine, 0.0) / (cosine * cosine);
        highp float inner2 = max(1.0 / (innerCosine * innerCosine) - 1.0, 0.000001);
        highp float outer2 = max(1.0 / (outerCosine * outerCosine) - 1.0, inner2 + 0.000001);
        highp float hotspot = 1.0 - smoothstep(0.0, inner2, radius2);
        highp float spill = 1.0 - smoothstep(inner2, outer2, radius2);
        cone = 0.65 * hotspot + 0.35 * spill;
    }
    return cone;
}

highp float flashlightDistance(highp float distance, highp float range) {
    // Fade the last quarter of the reach instead of clipping a bright disc
    // against walls at the range boundary.
    highp float fade = 1.0 - smoothstep(range * 0.75, range, distance);
    return fade / (1.0 + 0.1 * distance + 0.01 * distance * distance);
}


        out vec4 fragColor;

        in vec2 texCoord;
        in highp vec2 lightMapTexCoord;
        in highp vec4 atlasCoord;
        in highp vec3 worldPos;
        in vec3 worldNormal;

        // Material properties
        uniform sampler2D texture1; // lightmap
        uniform sampler2D texture2; // diffuse texture
        uniform vec3 ambientColor;  // authored mission-wide minimum lighting
        uniform float lightmapIntensity;
        uniform highp vec2 diffuseUvOffset;
        uniform bool unlit;
        uniform vec4 layerColorAlpha;
        uniform bool layerEnabled;
        uniform bool layerAlphaTest;
        // Polished-deck reflections; 0 is off. worldEnvironment says whether
        // the mission's capture is bound to shineEnvironment.
        uniform float worldGloss;
        uniform bool worldEnvironment;

        // Spotlight array uniforms (up to 6 spotlights)
        uniform highp vec3 spotlightPos[6];
        uniform vec4 spotlightColorIntensity[6];  // RGB + intensity
        uniform highp vec3 spotlightDirection[6];
        uniform highp float spotlightInnerAngle[6];
        uniform highp float spotlightOuterAngle[6];
        uniform highp float spotlightRange[6];
        // Slots 0..spotlightCount may hold a light; the rest are empty.
        uniform float spotlightCount;

        // Calculate spotlight contribution
        // One return: the Xbox 360's shader compiler inlines only single-exit
        // functions. Each early-out is a branch around the rest instead.
        vec3 calculateSpotlight(int i, highp vec3 worldPos, vec3 normal, vec3 texColor, inout vec3 specular, float power) {
            vec3 lit = vec3(0.0);
            highp vec3 lightVec = spotlightPos[i] - worldPos;
            highp float distance = length(lightVec);
            // Skip a light with zero intensity, or out of range.
            if (spotlightColorIntensity[i].w > 0.0 && !(distance > spotlightRange[i])) {
                highp vec3 lightDir = lightVec / max(distance, 0.00001);

                // Cone attenuation. A negative inner angle marks a point light -
                // it has no cone, so it lights every direction equally.
                highp float coneAttenuation = 1.0;
                if (spotlightInnerAngle[i] >= 0.0) {
                    highp float spotFactor = dot(-lightDir, normalize(spotlightDirection[i]));
                    coneAttenuation = flashlightCone(spotFactor, cos(spotlightInnerAngle[i]), cos(spotlightOuterAngle[i]));
                }

                // Distance attenuation
                highp float distanceAttenuation = flashlightDistance(distance, spotlightRange[i]);

                // Diffuse lighting
                float lambertian = max(dot(normal, lightDir), 0.0);

                if (shineHighlights()) {
                    vec3 radiance = spotlightColorIntensity[i].rgb * spotlightColorIntensity[i].w
                        * coneAttenuation * distanceAttenuation;
                    specular += shineHighlight(radiance, lightDir, normal, worldPos, power);
                }
                // Combine all factors
                lit = texColor * spotlightColorIntensity[i].rgb * spotlightColorIntensity[i].w
                       * lambertian * coneAttenuation * distanceAttenuation;
            }
            return lit;
        }

        void main() {
            // Sample lightmap and diffuse texture with proper UV wrapping
            float half_pixel = 0.5 / 4096.0;
            float full_pixel = half_pixel * 2.0;
            vec2 wrappedTexCoord = vec2(0.0, 0.0);
            float width = atlasCoord.z - full_pixel;
            float height = atlasCoord.w - full_pixel;

            wrappedTexCoord.x = mod(lightMapTexCoord.x * width, width) + atlasCoord.x + half_pixel;
            wrappedTexCoord.y = mod(lightMapTexCoord.y * height, height) + atlasCoord.y + half_pixel;

            vec4 lightmapColor = texture(texture1, wrappedTexCoord) * lightmapIntensity;
            vec4 diffuseColor = texture(texture2, texCoord + diffuseUvOffset);
            // Before any discard: the bump takes screen-space derivatives.
            vec3 geometric = normalize(worldNormal);
            vec3 bumped = unlit ? geometric
                : bumpNormal(texture2, texCoord + diffuseUvOffset, worldPos, geometric, 0.04);
            if (layerAlphaTest && diffuseColor.a * layerColorAlpha.a < 0.1) discard;

            // Dark's mission ambient is a minimum final intensity: preserve
            // brighter authored pixels while keeping unlit surfaces legible.
            vec3 finalColor = max(diffuseColor.rgb * lightmapColor.rgb, ambientColor);

            vec3 normal = geometric;
            // The Xbox 360's GPU takes about 10 ms a draw over these lamp
            // loops, even with each one's uniform off: it has no lamp
            // shadows, bump or gloss.
#ifndef SHOCK_XBOX360
            // An object between a shadow lamp and this surface takes back the
            // share of the baked light that lamp would have put here. World
            // casters are not in this set: the lightmap holds their shadows.
            if (!unlit) {
                highp float baked = max(dot(finalColor, vec3(0.299, 0.587, 0.114)), 0.001);
                // Each loop here sums into a local of its own: xost can't carry
                // into a hardware loop a value an if on a uniform changed.
                highp float kept = 1.0;
                for (int k = 0; k < SHOCK_COUNT(POINT_SHADOW_MAX); k++) {
                    highp vec4 lamp = pointShadowLights[k];
                    if (lamp.w <= 0.0) continue;
                    highp vec3 toLamp = lamp.xyz - worldPos;
                    highp float distance = length(toLamp);
                    if (distance >= lamp.w) continue;
                    float ndl = max(dot(normal, toLamp / max(distance, 0.00001)), 0.0);
                    if (ndl <= 0.0) continue;
                    float shadow = 1.0 - pointShadowVisibility(k, 1, worldPos, normal);
                    if (shadow <= 0.0) continue;
                    vec3 lampLight = diffuseColor.rgb * pointShadowColor[k].rgb * pointShadowColor[k].w
                        * ndl / max(distance, 0.8);
                    // The objects' lamp scale runs dimmer than the baked
                    // light it stands for; doubling the ratio lets a shadow
                    // take back what a player expects (a dark, readable shape).
                    highp float share = clamp(2.0 * dot(lampLight, vec3(0.299, 0.587, 0.114)) / baked, 0.0, 1.0);
                    kept *= 1.0 - pointShadowStrength * shadow * share;
                }
                finalColor = max(finalColor * kept, ambientColor);
            }

            // The lightmap was baked for the flat face; relight the relief
            // against the nearby lamps (and faintly from the eye, so an unlit
            // corridor still shows its texture) by how much the bumped normal
            // turns toward or away from them.
            if (!unlit && bumpStrength > 0.0) {
                highp vec3 toEyeDir = normalize(fogEyeDensity.xyz - worldPos);
                vec3 lightDir = toEyeDir * 0.35;
                for (int k = 0; k < SHOCK_COUNT(POINT_SHADOW_MAX); k++) {
                    highp vec4 lamp = pointShadowLights[k];
                    if (lamp.w <= 0.0) continue;
                    highp vec3 toLamp = lamp.xyz - worldPos;
                    highp float distance = length(toLamp);
                    if (distance >= lamp.w) continue;
                    lightDir += toLamp / max(distance, 0.00001) * (1.0 - distance / lamp.w);
                }
                float weight = length(lightDir);
                if (weight > 1e-3) {
                    vec3 l = lightDir / weight;
                    float relief = 1.0 + 1.6 * (dot(bumped, l) - dot(normal, l));
                    finalColor *= clamp(relief, 0.5, 1.5);
                }
            }

            // Reflections. Decks are polished metal and tile: floors most of
            // all, walls a little. The capture is a lit photograph of the
            // deck, so it is scaled by the light this spot actually gets -
            // a dark corner reflects nothing. Nearby lamps add sharp
            // highlights, which is what reads as a reflective floor.
            if (!unlit && worldGloss > 0.0) {
                highp vec3 toEye = normalize(fogEyeDensity.xyz - worldPos);
                float up = smoothstep(0.5, 0.9, geometric.y);
                float gloss = worldGloss * mix(0.08, 1.0, up);
                float facing = clamp(dot(bumped, toEye), 0.0, 1.0);
                float fresnel = mix(0.12, 1.0, pow(1.0 - facing, 5.0));
                float lit = clamp(dot(lightmapColor.rgb, vec3(0.299, 0.587, 0.114)) * 3.0, 0.0, 1.0);
                vec3 reflected = vec3(0.0);
                if (worldEnvironment) {
                    vec3 mirrored = reflect(-toEye, bumped);
                    reflected += textureLod(shineEnvironment, environmentDirection(mirrored), 1.2).rgb
                        * fresnel * lit;
                }
                vec3 lampGloss = vec3(0.0);
                for (int k = 0; k < SHOCK_COUNT(POINT_SHADOW_MAX); k++) {
                    highp vec4 lamp = pointShadowLights[k];
                    if (lamp.w <= 0.0) continue;
                    highp vec3 toLamp = lamp.xyz - worldPos;
                    highp float distance = length(toLamp);
                    if (distance >= lamp.w) continue;
                    vec3 l = toLamp / max(distance, 0.00001);
                    float ndl = max(dot(bumped, l), 0.0);
                    float spec = pow(max(dot(bumped, normalize(l + toEye)), 0.0), 72.0) * ndl;
                    float falloff = 1.0 - distance / lamp.w;
                    lampGloss += pointShadowColor[k].rgb * pointShadowColor[k].w
                        * spec * falloff * (0.6 + 2.0 * fresnel);
                }
                reflected += lampGloss;
                finalColor += reflected * gloss;
            }

#endif

            // Add dynamic spotlight contributions on top of baked lighting
            vec3 specular = vec3(0.0);
            float power = shineHighlights() ? shinePower(length(worldNormal)) : SHINE_POWER;
            vec3 spotlights = vec3(0.0);
            // Only the slots in use: the Xbox 360's shader compiler runs every
            // branch of a light, so each pass costs a full-screen layer about
            // 10 ms there, lit or not (vulkan4xenon's vkhang, 2026-11-26).

            finalColor += spotlights;

            vec4 base = vec4((unlit ? diffuseColor.rgb : finalColor) * layerColorAlpha.rgb,
                layerEnabled ? diffuseColor.a * layerColorAlpha.a : 1.0);
            // Terrain enables only the light-driven highlight (zero authored
            // sheen passes and no environment reflection). Reuse the object's
            // mask/coverage composition and move that mask with the diffuse.
            fragColor = applyShine(base, vec3(0.0), specular, 1.0,
                texCoord + diffuseUvOffset, worldPos, normal);
            if (!unlit) fragColor.rgb *= fogFactor(worldPos);
        }
