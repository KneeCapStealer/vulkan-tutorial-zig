const std = @import("std");
const mem = std.mem;
const meta = std.meta;
const builtin = @import("builtin");
const assert = std.debug.assert;

const glfw = @import("glfw");
const vk = @import("vulkan");
const img = @import("zigimg");

const math = @import("math.zig");
const Vec2 = math.Vec2;
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;

const Vertex = struct {
    pos: Vec3,
    color: Vec3,
    tex_coord: Vec2,

    fn getBindingDescription() vk.VertexInputBindingDescription {
        const binding_description: vk.VertexInputBindingDescription = .{
            .binding = 0,
            .stride = @sizeOf(Vertex),
            .input_rate = .vertex,
        };

        return binding_description;
    }

    fn getAttributeDescriptions() [3]vk.VertexInputAttributeDescription {
        return [_]vk.VertexInputAttributeDescription{ .{
            .binding = 0,
            .location = 0,
            .format = .r32g32b32_sfloat,
            .offset = @offsetOf(Vertex, "pos"),
        }, .{
            .binding = 0,
            .location = 1,
            .format = .r32g32b32_sfloat,
            .offset = @offsetOf(Vertex, "color"),
        }, .{
            .binding = 0,
            .location = 2,
            .format = .r32g32_sfloat,
            .offset = @offsetOf(Vertex, "tex_coord"),
        } };
    }
};

const UniformBufferObject = extern struct {
    model: Mat4 align(16),
    view: Mat4 align(16),
    proj: Mat4 align(16),
};

const App = @This();

var debug_allocator = if (builtin.mode == .Debug) std.heap.DebugAllocator(.{}){} else {};
const allocator = if (builtin.mode == .Debug) debug_allocator.allocator() else std.heap.smp_allocator;

const window_width = 800;
const window_height = 600;

// Validation layers
const enable_val_layers = builtin.mode == .Debug;
/// Layer names to load if validation layers are active
const val_layers: []const [*:0]const u8 = &.{
    "VK_LAYER_KHRONOS_validation",
};

const device_extensions: []const [*:0]const u8 = &.{
    vk.extensions.khr_swapchain.name,
};

const max_frames_in_flight = 2;

window: *glfw.Window,
surface: vk.SurfaceKHR,

vk_base: vk.BaseWrapper,
instance_wrapper: vk.InstanceWrapper,
vk_instance: vk.InstanceProxy,

vk_physical: vk.PhysicalDevice,
device_wrapper: vk.DeviceWrapper,
vk_device: vk.DeviceProxy,

graphics_queue: vk.QueueProxy,
present_queue: vk.QueueProxy,

debug_messenger: vk.DebugUtilsMessengerEXT,

swap_chain: vk.SwapchainKHR,
swap_chain_images: []vk.Image,
swap_chain_image_format: vk.Format,
swap_chain_extent: vk.Extent2D,
swap_chain_image_views: []vk.ImageView,
swap_chain_framebuffers: []vk.Framebuffer,

render_pass: vk.RenderPass,
descriptor_set_layout: vk.DescriptorSetLayout,
pipeline_layout: vk.PipelineLayout,
graphics_pipeline: vk.Pipeline,

command_pool: vk.CommandPool,
command_buffers: [max_frames_in_flight]vk.CommandBuffer,

image_available_semaphores: [max_frames_in_flight]vk.Semaphore,
render_finished_semaphores: [max_frames_in_flight]vk.Semaphore,
in_flight_fences: [max_frames_in_flight]vk.Fence,

current_frame: u32,

framebuffer_resized: bool,

verticies: []const Vertex,
vertex_buffer: vk.Buffer,
vertex_buffer_memory: vk.DeviceMemory,

indices: []const u32,
index_buffer: vk.Buffer,
index_buffer_memory: vk.DeviceMemory,

uniform_buffers: []vk.Buffer,
uniform_buffer_memories: []vk.DeviceMemory,
uniform_buffers_mapped: []*anyopaque,

descriptor_pool: vk.DescriptorPool,
descriptor_sets: []vk.DescriptorSet,

texture_image: vk.Image,
texture_image_memory: vk.DeviceMemory,
texture_image_view: vk.ImageView,
texture_sampler: vk.Sampler,

depth_image: vk.Image,
depth_image_memory: vk.DeviceMemory,
depth_image_view: vk.ImageView,

pub fn init() App {
    comptime var command_buffers: [max_frames_in_flight]vk.CommandBuffer = undefined;
    inline for (&command_buffers) |*buffer| {
        buffer.* = .null_handle;
    }

    comptime var image_available_semaphores: [max_frames_in_flight]vk.Semaphore = undefined;
    inline for (&image_available_semaphores) |*semaphore| {
        semaphore.* = .null_handle;
    }

    comptime var render_finished_semaphores: [max_frames_in_flight]vk.Semaphore = undefined;
    inline for (&render_finished_semaphores) |*semaphore| {
        semaphore.* = .null_handle;
    }

    comptime var in_flight_fences: [max_frames_in_flight]vk.Fence = undefined;
    inline for (&in_flight_fences) |*fence| {
        fence.* = .null_handle;
    }

    return App{
        .window = undefined,
        .surface = .null_handle,

        .vk_base = undefined,
        .instance_wrapper = undefined,
        .vk_instance = undefined,

        .vk_physical = .null_handle,
        .device_wrapper = undefined,
        .vk_device = undefined,

        .graphics_queue = undefined,
        .present_queue = undefined,

        .debug_messenger = .null_handle,

        .swap_chain = .null_handle,
        .swap_chain_images = undefined,
        .swap_chain_extent = undefined,
        .swap_chain_image_format = undefined,
        .swap_chain_image_views = undefined,
        .swap_chain_framebuffers = undefined,

        .render_pass = .null_handle,
        .descriptor_set_layout = .null_handle,
        .pipeline_layout = .null_handle,
        .graphics_pipeline = .null_handle,

        .command_pool = .null_handle,
        .command_buffers = command_buffers,

        .image_available_semaphores = image_available_semaphores,
        .render_finished_semaphores = render_finished_semaphores,
        .in_flight_fences = in_flight_fences,

        .current_frame = 0,

        .framebuffer_resized = false,

        .verticies = &.{
            Vertex{ .pos = .{ .x = -0.5, .y = -0.5, .z = 0.5 }, .color = .{ .x = 1, .y = 0, .z = 0 }, .tex_coord = .{ .x = 1, .y = 0 } },
            Vertex{ .pos = .{ .x = 0.5, .y = -0.5, .z = 0.5 }, .color = .{ .x = 0, .y = 1, .z = 0 }, .tex_coord = .{ .x = 0, .y = 0 } },
            Vertex{ .pos = .{ .x = 0.5, .y = 0.5, .z = 0.5 }, .color = .{ .x = 0, .y = 0, .z = 1 }, .tex_coord = .{ .x = 0, .y = 1 } },
            Vertex{ .pos = .{ .x = -0.5, .y = 0.5, .z = 0.5 }, .color = .{ .x = 1, .y = 1, .z = 1 }, .tex_coord = .{ .x = 1, .y = 1 } },

            Vertex{ .pos = .{ .x = -0.5, .y = -0.5, .z = 0 }, .color = .{ .x = 1, .y = 0, .z = 0 }, .tex_coord = .{ .x = 1, .y = 0 } },
            Vertex{ .pos = .{ .x = 0.5, .y = -0.5, .z = 0 }, .color = .{ .x = 0, .y = 1, .z = 0 }, .tex_coord = .{ .x = 0, .y = 0 } },
            Vertex{ .pos = .{ .x = 0.5, .y = 0.5, .z = 0 }, .color = .{ .x = 0, .y = 0, .z = 1 }, .tex_coord = .{ .x = 0, .y = 1 } },
            Vertex{ .pos = .{ .x = -0.5, .y = 0.5, .z = 0 }, .color = .{ .x = 1, .y = 1, .z = 1 }, .tex_coord = .{ .x = 1, .y = 1 } },
        },
        .vertex_buffer = .null_handle,
        .vertex_buffer_memory = .null_handle,

        .indices = &.{
            0, 1, 2, 0, 2, 3,
            4, 5, 6, 4, 6, 7,
        },
        .index_buffer = .null_handle,
        .index_buffer_memory = .null_handle,

        .uniform_buffers = &.{},
        .uniform_buffer_memories = &.{},
        .uniform_buffers_mapped = &.{},

        .descriptor_pool = .null_handle,
        .descriptor_sets = &.{},

        .texture_image = .null_handle,
        .texture_image_memory = .null_handle,
        .texture_image_view = .null_handle,
        .texture_sampler = .null_handle,

        .depth_image = .null_handle,
        .depth_image_memory = .null_handle,
        .depth_image_view = .null_handle,
    };
}

pub fn run(self: *App) !void {
    try glfw.init();

    try self.initWindow();
    try self.initVulkan();
    try self.mainLoop();
    self.cleanup();
}

fn initWindow(self: *App) !void {
    glfw.windowHint(glfw.WindowHint.client_api, glfw.ClientApi.no_api);

    self.window = try glfw.createWindow(window_width, window_height, "Vulkan", null);
    glfw.setWindowUserPointer(self.window, self);
    _ = glfw.setFramebufferSizeCallback(self.window, framebufferResizeCallback);
}

fn framebufferResizeCallback(window: *glfw.Window, _: c_int, _: c_int) callconv(.c) void {
    const self: *App = glfw.getWindowUserPointer(window, App).?;

    self.framebuffer_resized = true;
}

fn createInstance(self: *App) !void {
    if (enable_val_layers and !try self.checkValLayerSupport()) {
        return error.ValidationLayersNotAvailable;
    }

    // Validation layers:
    const enabled_layer_count: u32, const enabled_layer_names: ?[*]const [*:0]const u8 = if (enable_val_layers)
        .{ val_layers.len, val_layers.ptr }
    else
        .{ 0, null };

    const appInfo: vk.ApplicationInfo = .{
        .p_application_name = "Hello Triangle",
        .p_engine_name = "No Engine",
        .engine_version = @bitCast(vk.makeApiVersion(1, 0, 0, 0)),
        .application_version = @bitCast(vk.makeApiVersion(1, 0, 0, 0)),
        .api_version = @bitCast(vk.API_VERSION_1_0),
    };

    // extensions
    const extensions = try getRequiredExtensions();
    defer allocator.free(extensions);
    const instance_create_info: vk.InstanceCreateInfo = .{
        .p_application_info = &appInfo,
        .enabled_layer_count = enabled_layer_count,
        .pp_enabled_layer_names = enabled_layer_names,
        .enabled_extension_count = @intCast(extensions.len),
        .pp_enabled_extension_names = extensions.ptr,
        .p_next = if (enable_val_layers) &default_debug_messenger_create_info else null,
    };
    const instance = try self.vk_base.createInstance(&instance_create_info, null);
    self.instance_wrapper = .load(instance, self.vk_base.dispatch.vkGetInstanceProcAddr.?);
    self.vk_instance = .init(instance, &self.instance_wrapper);
}

fn createSurface(self: *App) !void {
    try glfw.createWindowSurface(@ptrFromInt(@intFromEnum(self.vk_instance.handle)), self.window, null, @ptrCast(&self.surface));
}

fn initVulkan(self: *App) !void {
    self.vk_base = .load(glfwGetInstanceProcAddress);

    try self.createInstance();
    try self.setupDebugMessenger();
    try self.createSurface();
    try self.pickPhysicalDevice();
    try self.createLogicalDevice();
    try self.createSwapChain();
    try self.createImageViews();
    try self.createRenderPass();
    try self.createDescriptorSetLayout();
    try self.createGraphicsPipeline();
    try self.createCommandPool();
    try self.createDepthResources();
    try self.createFramebuffers();
    try self.createTextureImage();
    try self.createTextureImageView();
    try self.createTextureSampler();
    try self.createVertexBuffer();
    try self.createIndexBuffer();
    try self.createUniformBuffers();
    try self.createDescriptorPool();
    try self.createDescriptorSets();
    try self.createCommandBuffers();
    try self.createSyncObjects();
}

fn findSupportedFormat(self: *App, candidates: []const vk.Format, tiling: vk.ImageTiling, features: vk.FormatFeatureFlags) !vk.Format {
    for (candidates) |format| {
        const props = self.vk_instance.getPhysicalDeviceFormatProperties(self.vk_physical, format);

        switch (tiling) {
            .linear => {
                if (props.linear_tiling_features.contains(features))
                    return format;
            },
            .optimal => {
                if (props.optimal_tiling_features.contains(features))
                    return format;
            },
            .drm_format_modifier_ext => {},
            _ => {},
        }
    }

    return error.FormatNotFound;
}

fn findDepthFormat(self: *App) !vk.Format {
    return self.findSupportedFormat(&.{
        .d32_sfloat,
        .d32_sfloat_s8_uint,
        .d24_unorm_s8_uint,
    }, .optimal, .{ .depth_stencil_attachment_bit = true });
}

fn hasStencilComponent(format: vk.Format) bool {
    return format == .d32_sfloat_s8_uint or format == .d24_unorm_s8_uint;
}

fn createDepthResources(self: *App) !void {
    const depth_format = try self.findDepthFormat();

    self.depth_image, self.depth_image_memory = try self.createImage(
        self.swap_chain_extent.width,
        self.swap_chain_extent.height,
        depth_format,
        .optimal,
        .{ .depth_stencil_attachment_bit = true },
        .{ .device_local_bit = true },
    );

    self.depth_image_view = try self.createImageView(
        self.depth_image,
        depth_format,
        .{ .depth_bit = true },
    );
}

fn createTextureSampler(self: *App) !void {
    const physical_device_properties = self.vk_instance.getPhysicalDeviceProperties(self.vk_physical);
    const sampler_info: vk.SamplerCreateInfo = .{
        // Bilinear vs no filtering
        // Basically if we want to just use each pixel in the texture
        // or if we want to blur between the pixels if there are not enough
        .mag_filter = .linear,
        .min_filter = .linear,
        // UV texture addressing move pr. axis.
        // Repeat the texture, or mirror repeat the texture and so on...
        // x axis
        .address_mode_u = .repeat,
        // y axis
        .address_mode_v = .repeat,
        // z axis
        .address_mode_w = .repeat,
        .anisotropy_enable = .true,
        .max_anisotropy = physical_device_properties.limits.max_sampler_anisotropy,
        .border_color = .int_opaque_black,
        // Made texture cordinates go in range: [0, 1], instead of [0, texWidth] and [0, texHeight]
        .unnormalized_coordinates = .false,
        // Can be used for shadowmap antialiasing
        .compare_enable = .false,
        .compare_op = .always,
        .mipmap_mode = .linear,
        .mip_lod_bias = 0,
        .max_lod = 0,
        .min_lod = 0,
    };

    self.texture_sampler = try self.vk_device.createSampler(&sampler_info, null);
}

fn createImageView(self: *App, image: vk.Image, format: vk.Format, aspect_flags: vk.ImageAspectFlags) !vk.ImageView {
    const view_info: vk.ImageViewCreateInfo = .{
        .image = image,
        .format = format,
        .view_type = .@"2d",
        .components = .{ .a = .identity, .b = .identity, .g = .identity, .r = .identity },
        .subresource_range = .{
            .aspect_mask = aspect_flags,
            .base_mip_level = 0,
            .base_array_layer = 0,
            .layer_count = 1,
            .level_count = 1,
        },
    };

    return try self.vk_device.createImageView(&view_info, null);
}

fn createTextureImageView(self: *App) !void {
    self.texture_image_view = try self.createImageView(self.texture_image, .r8g8b8a8_srgb, .{ .color_bit = true });
}

fn copyBufferToImage(self: *App, buffer: vk.Buffer, image: vk.Image, width: u32, height: u32) !void {
    const command_buffer = try self.beginSingleTimeCommands();
    const cmd: vk.CommandBufferProxy = .init(command_buffer, &self.device_wrapper);

    const region: vk.BufferImageCopy = .{ .buffer_offset = 0, .buffer_row_length = 0, .buffer_image_height = 0, .image_subresource = .{
        .mip_level = 0,
        .layer_count = 1,
        .aspect_mask = .{ .color_bit = true },
        .base_array_layer = 0,
    }, .image_offset = .{ .x = 0, .y = 0, .z = 0 }, .image_extent = .{
        .width = width,
        .height = height,
        .depth = 1,
    } };

    cmd.copyBufferToImage(buffer, image, .transfer_dst_optimal, 1, @ptrCast(&region));

    try self.endSingleTimeCommands(command_buffer);
}

fn transitionImageLayout(self: *App, image: vk.Image, format: vk.Format, old_layout: vk.ImageLayout, new_layout: vk.ImageLayout) !void {
    _ = format;

    const command_buffer = try self.beginSingleTimeCommands();
    const cmd: vk.CommandBufferProxy = .init(command_buffer, &self.device_wrapper);

    const src_access_mask, const dst_access_mask, const src_stage, const dst_stage = blk: {
        if (old_layout == .undefined and new_layout == .transfer_dst_optimal) {
            break :blk .{
                vk.AccessFlags{}, // non specific access
                vk.AccessFlags{ .transfer_write_bit = true }, // The access needs to have transfer write, as it will be used for transfer destination
                vk.PipelineStageFlags{ .top_of_pipe_bit = true }, // The pipeline stage is top, because it hasn't been used before
                vk.PipelineStageFlags{ .transfer_bit = true }, // The destination pipeline stage is transfer
            };
        } else if (old_layout == .transfer_dst_optimal and new_layout == .shader_read_only_optimal) {
            break :blk .{
                vk.AccessFlags{ .transfer_write_bit = true }, // The access was transfer write, as the image has just been written to
                vk.AccessFlags{ .shader_read_bit = true }, // The access flags become shader read, as the image will be used in fragment shader
                vk.PipelineStageFlags{ .transfer_bit = true }, // The pipeline has just been in the transfer stage
                vk.PipelineStageFlags{ .fragment_shader_bit = true }, // The image will next be used during the fragment shader
            };
        } else return error.InvalidLayoutTransition;
    };

    const barrier: vk.ImageMemoryBarrier = .{
        .old_layout = old_layout,
        .new_layout = new_layout,
        .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresource_range = .{
            .aspect_mask = .{ .color_bit = true },
            .base_mip_level = 0,
            .base_array_layer = 0,
            .layer_count = 1,
            .level_count = 1,
        },
        .src_access_mask = src_access_mask,
        .dst_access_mask = dst_access_mask,
    };

    cmd.pipelineBarrier(src_stage, dst_stage, .{}, 0, null, 0, null, 1, @ptrCast(&barrier));

    try self.endSingleTimeCommands(command_buffer);
}

fn beginSingleTimeCommands(self: *App) !vk.CommandBuffer {
    const alloc_info: vk.CommandBufferAllocateInfo = .{
        .command_buffer_count = 1,
        .command_pool = self.command_pool,
        .level = .primary,
    };

    var command_buffer: vk.CommandBuffer = undefined;
    try self.vk_device.allocateCommandBuffers(&alloc_info, @ptrCast(&command_buffer));

    const begin_info: vk.CommandBufferBeginInfo = .{ .flags = .{ .one_time_submit_bit = true } };
    try self.vk_device.beginCommandBuffer(command_buffer, &begin_info);

    return command_buffer;
}

inline fn endSingleTimeCommands(self: *App, command_buffer: vk.CommandBuffer) !void {
    try self.vk_device.endCommandBuffer(command_buffer);

    const submit_info: vk.SubmitInfo = .{
        .command_buffer_count = 1,
        .p_command_buffers = @ptrCast(&command_buffer),
    };
    try self.graphics_queue.submit(1, @ptrCast(&submit_info), .null_handle);
    try self.graphics_queue.waitIdle();

    self.vk_device.freeCommandBuffers(self.command_pool, 1, @ptrCast(&command_buffer));
}

inline fn createTextureImage(self: *App) !void {
    var read_buffer: [img.io.DEFAULT_BUFFER_SIZE]u8 = undefined;
    const self_path = try std.fs.selfExeDirPathAlloc(allocator);
    const self_dir = try std.fs.openDirAbsolute(self_path, .{ .access_sub_paths = true });
    allocator.free(self_path);

    const image_path = try self_dir.realpathAlloc(allocator, "../images/WaltahBetter.jpg");

    std.debug.print("image path: {s}\n", .{image_path});

    var image: img.Image = try .fromFilePath(allocator, image_path, &read_buffer);
    defer image.deinit(allocator);
    allocator.free(image_path);

    try image.convert(allocator, .rgba32);
    const pixels = image.pixels.rgba32;
    const PixelType = @typeInfo(@TypeOf(pixels)).pointer.child;
    const size: vk.DeviceSize = pixels.len * @sizeOf(PixelType);

    const staging_buffer, const staging_memory = try self.createBuffer(size, .{ .transfer_src_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true });

    // Copy the pixel data
    const data = try self.vk_device.mapMemory(staging_memory, 0, size, .{});
    @memcpy(@as([*]PixelType, @ptrCast(data.?)), pixels);
    self.vk_device.unmapMemory(staging_memory);

    self.texture_image, self.texture_image_memory = try self.createImage(
        // Size of the image
        @intCast(image.width),
        @intCast(image.width),
        // The format of `pixels.rgba24`
        .r8g8b8a8_srgb,
        // Optimal means the memory is laid out so the shader has more effecient color access.
        // This makes it hard or impossible to access image data outside of shader, since the colors are laid out "randomly"
        // If you need to access image data use `.linear`
        .optimal,
        .{
            .transfer_dst_bit = true,
            .sampled_bit = true,
        },
        .{ .device_local_bit = true },
    );

    try self.transitionImageLayout(self.texture_image, .r8g8b8a8_srgb, .undefined, .transfer_dst_optimal);
    try self.copyBufferToImage(staging_buffer, self.texture_image, @intCast(image.width), @intCast(image.width));
    try self.transitionImageLayout(self.texture_image, .r8g8b8a8_srgb, .transfer_dst_optimal, .shader_read_only_optimal);

    self.vk_device.destroyBuffer(staging_buffer, null);
    self.vk_device.freeMemory(staging_memory, null);
}

fn createImage(
    self: *App,
    width: u32,
    height: u32,
    format: vk.Format,
    tiling: vk.ImageTiling,
    usage: vk.ImageUsageFlags,
    properties: vk.MemoryPropertyFlags,
) !struct { vk.Image, vk.DeviceMemory } {
    const image_info: vk.ImageCreateInfo = .{
        // 2D image
        .image_type = .@"2d",
        .extent = .{
            .width = width,
            .height = height,
            .depth = 1,
        },
        // No Mip mapping
        .mip_levels = 1,
        // Not really sure what this means
        .array_layers = 1,
        .format = format,
        .tiling = tiling,
        // The image isn't layed out optimally already, if it was we could tell Vulkan it was `preinitialized`
        .initial_layout = .undefined,
        .usage = usage,
        .sharing_mode = .exclusive,
        // not sure, something with images that are attachments
        .samples = .{ .@"1_bit" = true },
    };
    const image = try self.vk_device.createImage(&image_info, null);

    const mem_requirements = self.vk_device.getImageMemoryRequirements(image);
    const alloc_info: vk.MemoryAllocateInfo = .{
        .allocation_size = mem_requirements.size,
        .memory_type_index = try self.findMemoryType(mem_requirements.memory_type_bits, properties),
    };

    const memory = try self.vk_device.allocateMemory(&alloc_info, null);
    try self.vk_device.bindImageMemory(image, memory, 0);

    return .{ image, memory };
}

fn createDescriptorSets(self: *App) !void {
    const layouts: []const vk.DescriptorSetLayout = &.{ self.descriptor_set_layout, self.descriptor_set_layout };
    assert(layouts.len == max_frames_in_flight);

    const alloc_info: vk.DescriptorSetAllocateInfo = .{
        .descriptor_pool = self.descriptor_pool,
        .descriptor_set_count = max_frames_in_flight,
        .p_set_layouts = layouts.ptr,
    };

    self.descriptor_sets = try allocator.alloc(vk.DescriptorSet, max_frames_in_flight);
    try self.vk_device.allocateDescriptorSets(&alloc_info, self.descriptor_sets.ptr);

    for (0..max_frames_in_flight) |i| {
        const buffer_info: vk.DescriptorBufferInfo = .{
            .buffer = self.uniform_buffers[i],
            .offset = 0,
            .range = @sizeOf(UniformBufferObject),
        };
        const image_info: vk.DescriptorImageInfo = .{
            .image_layout = .shader_read_only_optimal,
            .image_view = self.texture_image_view,
            .sampler = self.texture_sampler,
        };
        const descriptor_writes: [2]vk.WriteDescriptorSet = .{
            .{
                .dst_set = self.descriptor_sets[i],
                .dst_binding = 0,
                .dst_array_element = 0,
                .descriptor_type = .uniform_buffer,
                .descriptor_count = 1,
                .p_buffer_info = @ptrCast(&buffer_info),
                // These are irelevant but don't accept nullptrs ???
                .p_image_info = @ptrFromInt(8),
                .p_texel_buffer_view = @ptrFromInt(8),
            },
            .{
                .dst_set = self.descriptor_sets[i],
                .dst_binding = 1,
                .dst_array_element = 0,
                .descriptor_type = .combined_image_sampler,
                .descriptor_count = 1,
                .p_buffer_info = undefined,
                // Now we are using images
                .p_image_info = @ptrCast(&image_info),
                .p_texel_buffer_view = undefined,
            },
        };

        self.vk_device.updateDescriptorSets(descriptor_writes.len, &descriptor_writes, 0, null);
    }
}

fn createDescriptorPool(self: *App) !void {
    const pool_sizes: [2]vk.DescriptorPoolSize = .{
        .{
            .descriptor_count = max_frames_in_flight,
            .type = .uniform_buffer,
        },
        .{
            .descriptor_count = max_frames_in_flight,
            .type = .combined_image_sampler,
        },
    };

    const pool_info: vk.DescriptorPoolCreateInfo = .{
        .pool_size_count = pool_sizes.len,
        .p_pool_sizes = &pool_sizes,
        .max_sets = max_frames_in_flight,
    };

    self.descriptor_pool = try self.vk_device.createDescriptorPool(&pool_info, null);
}

fn createUniformBuffers(self: *App) !void {
    const buffer_size: vk.DeviceSize = @sizeOf(UniformBufferObject);

    self.uniform_buffers = try allocator.alloc(vk.Buffer, max_frames_in_flight);
    self.uniform_buffer_memories = try allocator.alloc(vk.DeviceMemory, max_frames_in_flight);
    self.uniform_buffers_mapped = try allocator.alloc(*anyopaque, max_frames_in_flight);

    for (self.uniform_buffers, self.uniform_buffer_memories, self.uniform_buffers_mapped) |*buffer, *memory, *mapped| {
        buffer.*, memory.* = try self.createBuffer(buffer_size, .{ .uniform_buffer_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true });

        const possibly_null = try self.vk_device.mapMemory(memory.*, 0, buffer_size, vk.MemoryMapFlags.fromInt(0));
        mapped.* = possibly_null.?;
    }
}

fn createDescriptorSetLayout(self: *App) !void {
    const ubo_layout_binding: vk.DescriptorSetLayoutBinding = .{
        .binding = 0,
        .descriptor_type = .uniform_buffer,
        .descriptor_count = 1,
        .stage_flags = .{ .vertex_bit = true },
        .p_immutable_samplers = null,
    };

    const sampler_layout_binding: vk.DescriptorSetLayoutBinding = .{
        .binding = 1,
        .descriptor_type = .combined_image_sampler,
        .descriptor_count = 1,
        .stage_flags = .{ .fragment_bit = true },
        .p_immutable_samplers = null,
    };

    const bindings: []const vk.DescriptorSetLayoutBinding = &.{ ubo_layout_binding, sampler_layout_binding };

    const layout_info: vk.DescriptorSetLayoutCreateInfo = .{
        .binding_count = bindings.len,
        .p_bindings = bindings.ptr,
    };

    self.descriptor_set_layout = try self.vk_device.createDescriptorSetLayout(&layout_info, null);
}

fn createIndexBuffer(self: *App) !void {
    const buffer_size: vk.DeviceSize = @sizeOf(u32) * self.indices.len;

    // Create host visible buffer
    const staging_buffer, const staging_buffer_memory = try self.createBuffer(buffer_size, .{ .transfer_src_bit = true }, .{
        .host_visible_bit = true,
        .host_coherent_bit = true,
    });

    // Copy data to host visisble buffer
    const data = try self.vk_device.mapMemory(staging_buffer_memory, 0, buffer_size, .{});
    const ptr: [*]u32 = @ptrCast(@alignCast(data));
    @memcpy(ptr, self.indices);
    self.vk_device.unmapMemory(staging_buffer_memory);

    // Create a device local buffer and copy data
    self.index_buffer, self.index_buffer_memory = try self.createBuffer(buffer_size, .{
        .index_buffer_bit = true,
        .transfer_dst_bit = true,
    }, .{ .device_local_bit = true });
    try self.copyBuffer(staging_buffer, self.index_buffer, buffer_size);

    self.vk_device.destroyBuffer(staging_buffer, null);
    self.vk_device.freeMemory(staging_buffer_memory, null);
}

fn createVertexBuffer(self: *App) !void {
    const buffer_size: vk.DeviceSize = @sizeOf(Vertex) * self.verticies.len;
    const staging_buffer, const staging_buffer_memory = try self.createBuffer(buffer_size, .{ .transfer_src_bit = true }, .{
        .host_visible_bit = true,
        .host_coherent_bit = true,
    });

    const data = try self.vk_device.mapMemory(staging_buffer_memory, 0, buffer_size, .{});
    const ptr: [*]Vertex = @ptrCast(@alignCast(data));
    @memcpy(ptr, self.verticies);
    self.vk_device.unmapMemory(staging_buffer_memory);

    self.vertex_buffer, self.vertex_buffer_memory = try self.createBuffer(buffer_size, .{
        .vertex_buffer_bit = true,
        .transfer_dst_bit = true,
    }, .{ .device_local_bit = true });

    try self.copyBuffer(staging_buffer, self.vertex_buffer, buffer_size);

    self.vk_device.destroyBuffer(staging_buffer, null);
    self.vk_device.freeMemory(staging_buffer_memory, null);
}

fn copyBuffer(self: *App, src: vk.Buffer, dst: vk.Buffer, size: vk.DeviceSize) !void {
    const command_buffer = try self.beginSingleTimeCommands();
    const cmd: vk.CommandBufferProxy = .init(command_buffer, &self.device_wrapper);

    const copy_region: vk.BufferCopy = .{
        .dst_offset = 0,
        .src_offset = 0,
        .size = size,
    };
    cmd.copyBuffer(src, dst, 1, @ptrCast(&copy_region));

    try self.endSingleTimeCommands(command_buffer);
}

fn createBuffer(self: *App, size: vk.DeviceSize, usage: vk.BufferUsageFlags, properties: vk.MemoryPropertyFlags) !struct { vk.Buffer, vk.DeviceMemory } {
    const indices = try self.findQueueFamilies(self.vk_physical);
    const buffer_info: vk.BufferCreateInfo = .{
        .size = size,
        .usage = usage,
        .sharing_mode = .exclusive,
        .queue_family_index_count = 1,
        .p_queue_family_indices = &.{indices.graphics_family.?},
    };

    const buffer = try self.vk_device.createBuffer(&buffer_info, null);

    const mem_req = self.vk_device.getBufferMemoryRequirements(buffer);
    const alloc_info: vk.MemoryAllocateInfo = .{
        .allocation_size = mem_req.size,
        .memory_type_index = try self.findMemoryType(mem_req.memory_type_bits, properties),
    };

    const buffer_memory = try self.vk_device.allocateMemory(&alloc_info, null);

    try self.vk_device.bindBufferMemory(buffer, buffer_memory, 0);

    return .{ buffer, buffer_memory };
}

fn findMemoryType(self: *App, type_filter: u32, properties: vk.MemoryPropertyFlags) error{NoSuitableMemoryType}!u32 {
    const mem_properties = self.vk_instance.getPhysicalDeviceMemoryProperties(self.vk_physical);
    for (0..mem_properties.memory_type_count) |i| {
        const type_bit: u32 = @as(u32, 1) << @intCast(i);
        const memory_type_in_filter = type_filter & type_bit != 0;

        const memory_type_has_properties = mem_properties.memory_types[i].property_flags.contains(properties);

        if (memory_type_in_filter and memory_type_has_properties) {
            return @intCast(i);
        }
    }

    return error.NoSuitableMemoryType;
}

fn createSyncObjects(self: *App) !void {
    const semaphore_info: vk.SemaphoreCreateInfo = .{};
    const fence_info: vk.FenceCreateInfo = .{ .flags = .{ .signaled_bit = true } };

    inline for (0..max_frames_in_flight) |i| {
        self.image_available_semaphores[i] = try self.vk_device.createSemaphore(&semaphore_info, null);
        self.render_finished_semaphores[i] = try self.vk_device.createSemaphore(&semaphore_info, null);
        self.in_flight_fences[i] = try self.vk_device.createFence(&fence_info, null);
    }
}

fn recordCommandBuffer(self: *App, command_buffer: vk.CommandBuffer, image_index: u32) !void {
    const cmd_buf: vk.CommandBufferProxy = .init(command_buffer, self.vk_device.wrapper);

    const begin_info: vk.CommandBufferBeginInfo = .{};

    try cmd_buf.beginCommandBuffer(&begin_info);

    const clear_colors: [2]vk.ClearValue = .{
        .{ .color = .{ .float_32 = .{ 0, 0, 0, 1 } } },
        .{ .depth_stencil = .{ .depth = 1, .stencil = 0 } },
    };
    const render_pass_info: vk.RenderPassBeginInfo = .{
        .render_pass = self.render_pass,
        .framebuffer = self.swap_chain_framebuffers[image_index],
        .render_area = .{ .extent = self.swap_chain_extent, .offset = .{ .x = 0, .y = 0 } },
        .clear_value_count = @intCast(clear_colors.len),
        .p_clear_values = &clear_colors,
    };

    // inline means all commands will be in the primary command buffer
    cmd_buf.beginRenderPass(&render_pass_info, .@"inline");
    cmd_buf.bindPipeline(.graphics, self.graphics_pipeline);

    // Viewports and scissors are dynamic, hence we need to set them now
    const viewport: vk.Viewport = .{
        .x = 0,
        .y = 0,
        .width = @floatFromInt(self.swap_chain_extent.width),
        .height = @floatFromInt(self.swap_chain_extent.height),
        .min_depth = 0,
        .max_depth = 1,
    };
    const scissor: vk.Rect2D = .{
        .offset = .{ .x = 0, .y = 0 },
        .extent = self.swap_chain_extent,
    };

    cmd_buf.setViewport(0, 1, @ptrCast(&viewport));
    cmd_buf.setScissor(0, 1, @ptrCast(&scissor));

    // You can split color and position data into 2 diffrent buffers
    // And bind them all in different locations.
    // It can on some hardware improve performance to split up the position from texture map locations and normals.
    const vertex_buffers = [_]vk.Buffer{self.vertex_buffer};
    const offsets = [_]vk.DeviceSize{0};
    cmd_buf.bindVertexBuffers(0, 1, &vertex_buffers, &offsets);
    cmd_buf.bindIndexBuffer(self.index_buffer, 0, .uint32);

    cmd_buf.bindDescriptorSets(.graphics, self.pipeline_layout, 0, 1, @ptrCast(&self.descriptor_sets[self.current_frame]), 0, null);
    cmd_buf.drawIndexed(@intCast(self.indices.len), 1, 0, 0, 0);
    cmd_buf.endRenderPass();

    try cmd_buf.endCommandBuffer();
}

fn createCommandBuffers(self: *App) !void {
    const alloc_info: vk.CommandBufferAllocateInfo = .{
        .command_pool = self.command_pool,
        .command_buffer_count = max_frames_in_flight,
        .level = .primary,
    };
    try self.vk_device.allocateCommandBuffers(&alloc_info, &self.command_buffers);
}

fn createCommandPool(self: *App) !void {
    const queue_families = try self.findQueueFamilies(self.vk_physical);

    const pool_info: vk.CommandPoolCreateInfo = .{
        .flags = .{ .reset_command_buffer_bit = true },
        .queue_family_index = queue_families.graphics_family.?,
    };
    self.command_pool = try self.vk_device.createCommandPool(&pool_info, null);
}

fn createFramebuffers(self: *App) !void {
    self.swap_chain_framebuffers = try allocator.alloc(vk.Framebuffer, self.swap_chain_image_views.len);
    for (self.swap_chain_image_views, 0..) |image_view, i| {
        const attachments: [2]vk.ImageView = .{ image_view, self.depth_image_view };

        const framebuffer_info: vk.FramebufferCreateInfo = .{
            .render_pass = self.render_pass,
            // Specifies the attachments bound to the renderpass.
            // When the attachment at position 0 is written to, the image view at position 0 will be used
            .attachment_count = attachments.len,
            .p_attachments = &attachments,

            .width = self.swap_chain_extent.width,
            .height = self.swap_chain_extent.height,
            .layers = 1,
        };
        self.swap_chain_framebuffers[i] = try self.vk_device.createFramebuffer(&framebuffer_info, null);
    }
}

fn createRenderPass(self: *App) !void {
    const color_attachment: vk.AttachmentDescription = .{
        .format = self.swap_chain_image_format,
        .samples = .{ .@"1_bit" = true },
        .load_op = .clear,
        .store_op = .store,
        .stencil_load_op = .dont_care,
        .stencil_store_op = .dont_care,
        .initial_layout = .undefined,
        .final_layout = .present_src_khr,
    };

    const color_attachment_ref: vk.AttachmentReference = .{ .attachment = 0, .layout = .color_attachment_optimal };

    const depth_attachment: vk.AttachmentDescription = .{
        .format = try self.findDepthFormat(),
        .samples = .{ .@"1_bit" = true },
        .load_op = .clear,
        .store_op = .store,
        .initial_layout = .undefined,
        .final_layout = .depth_stencil_attachment_optimal,
        .stencil_load_op = .dont_care,
        .stencil_store_op = .dont_care,
    };

    const depth_attachment_ref: vk.AttachmentReference = .{ .attachment = 1, .layout = .depth_stencil_attachment_optimal };

    const subpass: vk.SubpassDescription = .{
        .pipeline_bind_point = .graphics,
        .color_attachment_count = 1,
        .p_color_attachments = @ptrCast(&color_attachment_ref),
        .p_depth_stencil_attachment = &depth_attachment_ref,
    };

    const dependency: vk.SubpassDependency = .{
        // No subpass is started yet
        .src_subpass = vk.SUBPASS_EXTERNAL,
        // The current subpass
        .dst_subpass = 0,
        // Wait for these stages to be completed / ready
        .src_stage_mask = .{
            .color_attachment_output_bit = true,
            .late_fragment_tests_bit = true,
        },
        .src_access_mask = .{ .depth_stencil_attachment_write_bit = true },

        // Before starting the color write and depth checking
        .dst_stage_mask = .{
            .color_attachment_output_bit = true,
            .early_fragment_tests_bit = true,
        },
        .dst_access_mask = .{
            .color_attachment_write_bit = true,
            .depth_stencil_attachment_write_bit = true,
        },
    };

    const attachments: [2]vk.AttachmentDescription = .{ color_attachment, depth_attachment };

    const render_pass_info: vk.RenderPassCreateInfo = .{
        .attachment_count = attachments.len,
        .p_attachments = &attachments,
        .subpass_count = 1,
        .p_subpasses = @ptrCast(&subpass),
        .dependency_count = 1,
        .p_dependencies = @ptrCast(&dependency),
    };

    self.render_pass = try self.vk_device.createRenderPass(&render_pass_info, null);
}

fn createGraphicsPipeline(self: *App) !void {
    const self_path = try std.fs.selfExeDirPathAlloc(allocator);
    std.debug.print("self path: {s}\n", .{self_path});
    const self_dir = try std.fs.openDirAbsolute(self_path, .{ .access_sub_paths = true });
    allocator.free(self_path);

    const vertex_shader = try self_dir.openFile("../shaders/out/vert.spv", std.fs.File.OpenFlags{
        .lock = .shared,
        .mode = .read_only,
    });

    errdefer vertex_shader.close();
    const fragment_shader = try self_dir.openFile("../shaders/out/frag.spv", std.fs.File.OpenFlags{
        .lock = .shared,
        .mode = .read_only,
    });
    errdefer fragment_shader.close();

    // Shader creation
    const vert_code = try vertex_shader.readToEndAlloc(allocator, std.math.maxInt(usize));
    defer allocator.free(vert_code);
    vertex_shader.close();
    const frag_code = try fragment_shader.readToEndAlloc(allocator, std.math.maxInt(usize));
    defer allocator.free(frag_code);
    fragment_shader.close();

    const vert_shader_module = try self.createShaderModule(vert_code);
    defer self.vk_device.destroyShaderModule(vert_shader_module, null);
    const frag_shader_module = try self.createShaderModule(frag_code);
    defer self.vk_device.destroyShaderModule(frag_shader_module, null);

    const vert_shader_stage_info: vk.PipelineShaderStageCreateInfo = .{
        .module = vert_shader_module,
        .stage = .{ .vertex_bit = true },
        .p_name = "main",
    };
    const frag_shader_stage_info: vk.PipelineShaderStageCreateInfo = .{
        .module = frag_shader_module,
        .stage = .{ .fragment_bit = true },
        .p_name = "main",
    };

    const shader_stages: [2]vk.PipelineShaderStageCreateInfo = .{ vert_shader_stage_info, frag_shader_stage_info };

    // States that are dynamic in the pipeline
    const dynamic_states: []const vk.DynamicState = &.{
        vk.DynamicState.viewport,
        vk.DynamicState.scissor,
    };

    const dynamic_state: vk.PipelineDynamicStateCreateInfo = .{
        .dynamic_state_count = dynamic_states.len,
        .p_dynamic_states = dynamic_states.ptr,
    };

    const vertex_binding_desc = Vertex.getBindingDescription();
    const vertex_attr_desc = Vertex.getAttributeDescriptions();
    const vertex_input_info: vk.PipelineVertexInputStateCreateInfo = .{
        .vertex_binding_description_count = 1,
        .p_vertex_binding_descriptions = @ptrCast(&vertex_binding_desc),
        .vertex_attribute_description_count = @intCast(vertex_attr_desc.len),
        .p_vertex_attribute_descriptions = &vertex_attr_desc,
    };

    // We are drawing triangles, not lines, or points
    const input_assembly: vk.PipelineInputAssemblyStateCreateInfo = .{ .topology = .triangle_list, .primitive_restart_enable = .false };

    const viewport: vk.Viewport = .{
        .width = @floatFromInt(self.swap_chain_extent.width),
        .height = @floatFromInt(self.swap_chain_extent.height),
        .x = 0,
        .y = 0,
        .max_depth = 1,
        .min_depth = 0,
    };

    const scissor: vk.Rect2D = .{ .extent = self.swap_chain_extent, .offset = .{ .y = 0, .x = 0 } };

    const viewport_state: vk.PipelineViewportStateCreateInfo = .{
        .viewport_count = 1,
        .p_viewports = @ptrCast(&viewport),
        .scissor_count = 1,
        .p_scissors = @ptrCast(&scissor),
    };

    const rasterizer: vk.PipelineRasterizationStateCreateInfo = .{
        .depth_clamp_enable = .false,
        .rasterizer_discard_enable = .false,
        .polygon_mode = .fill,
        .line_width = 1,
        .cull_mode = .{},
        .front_face = .counter_clockwise,
        .depth_bias_enable = .false,
        .depth_bias_clamp = 0,
        .depth_bias_constant_factor = 0,
        .depth_bias_slope_factor = 0,
    };

    const multisampling: vk.PipelineMultisampleStateCreateInfo = .{
        .sample_shading_enable = .false,
        .rasterization_samples = .{ .@"1_bit" = true },
        .min_sample_shading = 1,
        .alpha_to_coverage_enable = .false,
        .alpha_to_one_enable = .false,
    };

    const color_blend_attachment: vk.PipelineColorBlendAttachmentState = .{
        // Use all colors
        .color_write_mask = .{ .a_bit = true, .b_bit = true, .g_bit = true, .r_bit = true },

        // Don't blend colors with previos colors
        .blend_enable = .false,
        .src_color_blend_factor = .one,
        .dst_color_blend_factor = .zero,
        .color_blend_op = .add,
        .src_alpha_blend_factor = .one,
        .dst_alpha_blend_factor = .zero,
        .alpha_blend_op = .add,
    };

    const color_blend: vk.PipelineColorBlendStateCreateInfo = .{
        // Again don't do color blending
        .logic_op_enable = .false,
        .logic_op = .copy,
        .attachment_count = 1,
        .p_attachments = @ptrCast(&color_blend_attachment),
        .blend_constants = .{ 0.0, 0.0, 0.0, 0.0 },
    };

    const pipeline_layout_info: vk.PipelineLayoutCreateInfo = .{
        .set_layout_count = 1,
        .p_set_layouts = @ptrCast(@alignCast(&self.descriptor_set_layout)),
    };

    self.pipeline_layout = try self.vk_device.createPipelineLayout(&pipeline_layout_info, null);

    const depth_stencil: vk.PipelineDepthStencilStateCreateInfo = .{
        .depth_test_enable = .true,
        .depth_write_enable = .true,
        .depth_compare_op = .less,
        .depth_bounds_test_enable = .false,
        .max_depth_bounds = 1,
        .min_depth_bounds = 0,
        .stencil_test_enable = .false,
        // These are my random values that I hope just don't do anything because I have no clue how they work
        .back = .{
            .compare_mask = 0,
            .compare_op = .never,
            .depth_fail_op = .keep,
            .fail_op = .keep,
            .pass_op = .keep,
            .reference = 0,
            .write_mask = 0,
        },
        .front = .{
            .compare_mask = 0,
            .compare_op = .never,
            .depth_fail_op = .keep,
            .fail_op = .keep,
            .pass_op = .keep,
            .reference = 0,
            .write_mask = 0,
        },
    };

    const pipeline_info: vk.GraphicsPipelineCreateInfo = .{
        // Vertex and fragment shaders
        .stage_count = 2,
        .p_stages = &shader_stages,
        .p_vertex_input_state = &vertex_input_info,
        .p_input_assembly_state = &input_assembly,
        .p_viewport_state = &viewport_state,
        .p_rasterization_state = &rasterizer,
        .p_multisample_state = &multisampling,
        .p_depth_stencil_state = &depth_stencil,
        .p_color_blend_state = &color_blend,
        .p_dynamic_state = &dynamic_state,
        .layout = self.pipeline_layout,
        .render_pass = self.render_pass,
        .subpass = 0,
        .base_pipeline_handle = .null_handle,
        .base_pipeline_index = -1,
    };

    _ = try self.vk_device.createGraphicsPipelines(.null_handle, 1, @ptrCast(&pipeline_info), null, @ptrCast(&self.graphics_pipeline));
}

/// Takes in raw shader code, and spits out a shader module. !!(Lifetime of the code slice must match or outlive the shadermodule)!!
fn createShaderModule(self: *App, code: []const u8) !vk.ShaderModule {
    const create_info: vk.ShaderModuleCreateInfo = .{
        .code_size = code.len,
        .p_code = @ptrCast(@alignCast(code.ptr)),
    };

    return try self.vk_device.createShaderModule(&create_info, null);
}

fn createImageViews(self: *App) !void {
    self.swap_chain_image_views = try allocator.alloc(vk.ImageView, self.swap_chain_images.len);

    for (self.swap_chain_images, 0..) |image, i| {
        self.swap_chain_image_views[i] = try self.createImageView(image, self.swap_chain_image_format, .{ .color_bit = true });
    }
}

fn recreateSwapChain(self: *App) !void {
    // If the windows size is 0 then it is minimized and we pause the app
    var new_width: c_int = 0;
    var new_height: c_int = 0;
    glfw.getFramebufferSize(self.window, &new_width, &new_height);
    while (new_height == 0 or new_width == 0) {
        glfw.getFramebufferSize(self.window, &new_width, &new_height);
        glfw.waitEvents();
    }

    try self.vk_device.deviceWaitIdle();

    try self.cleanupSwapChain();

    try self.createSwapChain();
    try self.createImageViews();
    try self.createDepthResources();
    try self.createFramebuffers();
}

fn cleanupSwapChain(self: *App) !void {
    self.vk_device.destroyImageView(self.depth_image_view, null);
    self.vk_device.destroyImage(self.depth_image, null);
    self.vk_device.freeMemory(self.depth_image_memory, null);

    for (self.swap_chain_framebuffers) |framebuffer| {
        self.vk_device.destroyFramebuffer(framebuffer, null);
    }
    allocator.free(self.swap_chain_framebuffers);

    for (self.swap_chain_image_views) |view| {
        self.vk_device.destroyImageView(view, null);
    }
    allocator.free(self.swap_chain_image_views);
    allocator.free(self.swap_chain_images);

    self.vk_device.destroySwapchainKHR(self.swap_chain, null);
}

fn createSwapChain(self: *App) !void {
    const swap_chain_details = try self.querySwapChainSupport(self.vk_physical);
    defer allocator.free(swap_chain_details.formats);
    defer allocator.free(swap_chain_details.present_modes);

    // Find optimal settings for the swap chain.
    const format = self.chooseSwapSurfaceFormat(swap_chain_details.formats);
    const present_mode = self.chooseSwapPresentMode(swap_chain_details.present_modes);
    const extent = self.chooseSwapExtent(&swap_chain_details.capabilites);

    // Calculate numer of images in swapchain (probably 3 or 4).
    const requested_image_count = swap_chain_details.capabilites.min_image_count + 1;
    const image_count =
        if (swap_chain_details.capabilites.max_image_count == 0) requested_image_count else std.math.clamp(requested_image_count, swap_chain_details.capabilites.min_image_count, swap_chain_details.capabilites.max_image_count);

    // Get the queues to render to the swap chain images.
    const queue_families = try self.findQueueFamilies(self.vk_physical);
    const indices: [*]const u32 = &[_]u32{ queue_families.graphics_family.?, queue_families.present_family.? };

    // If we need 2 queues we need some different settings than if we only need 1 queue.
    const image_sharing_mode: vk.SharingMode, const queue_family_index_count: u32, const queue_family_indices: ?[*]const u32 =
        // Only 1 queue to use.
        if (queue_families.present_family == queue_families.graphics_family)
            .{ vk.SharingMode.exclusive, 0, null }
            // Use both queues.
        else
            .{ vk.SharingMode.concurrent, 2, indices };

    const swap_chain_create_info: vk.SwapchainCreateInfoKHR = .{
        .surface = self.surface,
        // All of the optimal settings we found previously.
        .min_image_count = image_count,
        .image_extent = extent,
        .present_mode = present_mode,
        .image_format = format.format,
        .image_color_space = format.color_space,
        // This is almost always the case.
        .image_array_layers = 1,
        // We will be rendering directly to the framebuffer.
        // Don't do this if you want to add post processing.
        .image_usage = .{ .color_attachment_bit = true },
        // Depending on hardware we may need to use multiple queues.
        // And to keep the tutorial simple we use concurrent mode for this, even though it is less performant.
        .image_sharing_mode = image_sharing_mode,
        .queue_family_index_count = queue_family_index_count,
        .p_queue_family_indices = queue_family_indices,
        // No transform applied.
        .pre_transform = swap_chain_details.capabilites.current_transform,
        // Don't blend the image with other windows or the background.
        .composite_alpha = .{ .opaque_bit_khr = true },
        // If pixels are obscured (another window is covering them), don't render their color.
        // This might be a bad option for some post processing effects, but it enabled better performance.
        .clipped = .true,
        // Since we don't have an old swapchain, this value is null
        // If we were recreating a swapchain (e.g. when resizing), this field MUST be specified
        .old_swapchain = vk.SwapchainKHR.null_handle,
    };

    self.swap_chain = try self.vk_device.createSwapchainKHR(&swap_chain_create_info, null);
    self.swap_chain_images = try self.vk_device.getSwapchainImagesAllocKHR(self.swap_chain, allocator);

    self.swap_chain_image_format = format.format;
    self.swap_chain_extent = extent;
}

/// Create logical device to encapsulate physical device
/// and it's queues
fn createLogicalDevice(self: *App) !void {
    const indices = try self.findQueueFamilies(self.vk_physical);
    if (!indices.isComplete()) {
        return error.FailedToFindQueueFamilies;
    }

    var unique_queue_families: std.ArrayList(u32) = .empty;
    defer unique_queue_families.deinit(allocator);

    const all_queue_families: []const u32 = &.{ indices.present_family.?, indices.graphics_family.? };
    outer: for (all_queue_families) |family| {
        for (unique_queue_families.items) |unique| {
            if (family == unique) {
                // The family is already in the ArrayList
                continue :outer;
            }
        }

        try unique_queue_families.append(allocator, family);
    }

    var queue_create_infos: std.ArrayList(vk.DeviceQueueCreateInfo) = .empty;
    defer queue_create_infos.deinit(allocator);

    const queue_prio: f32 = 1.0;

    for (unique_queue_families.items) |queue_family| {
        try queue_create_infos.append(allocator, vk.DeviceQueueCreateInfo{
            .queue_family_index = queue_family,
            .queue_count = 1,
            .p_queue_priorities = @ptrCast(&queue_prio),
        });
    }

    // Activate no features for now
    const device_features: vk.PhysicalDeviceFeatures = .{
        .sampler_anisotropy = .true,
    };
    const create_info: vk.DeviceCreateInfo = .{
        .p_queue_create_infos = queue_create_infos.items.ptr,
        .queue_create_info_count = @intCast(queue_create_infos.items.len),
        .p_enabled_features = &device_features,
        .pp_enabled_extension_names = device_extensions.ptr,
        .enabled_extension_count = device_extensions.len,
    };

    // Device creation
    const device = try self.vk_instance.createDevice(self.vk_physical, &create_info, null);
    self.device_wrapper = .load(device, self.instance_wrapper.dispatch.vkGetDeviceProcAddr.?);
    self.vk_device = .init(device, &self.device_wrapper);

    // Device Queue handles
    const graphics_queue = self.vk_device.getDeviceQueue(indices.graphics_family.?, 0);
    self.graphics_queue = .init(graphics_queue, &self.device_wrapper);

    const present_queue = self.vk_device.getDeviceQueue(indices.present_family.?, 0);
    self.present_queue = .init(present_queue, &self.device_wrapper);
}

fn pickPhysicalDevice(self: *App) !void {
    const physical_devices = try self.vk_instance.enumeratePhysicalDevicesAlloc(allocator);
    defer allocator.free(physical_devices);
    if (physical_devices.len == 0) {
        return error.NoVulkanDevice;
    }

    for (physical_devices) |device| {
        if (try self.isDeviceSuitable(device)) {
            self.vk_physical = device;
        }
    }

    if (self.vk_physical == .null_handle) {
        return error.NoSuitableDevice;
    }
}

/// Determines if a device can be used for the application
/// Returns true if the device has all the required features, and false if the device can't run the program
fn isDeviceSuitable(self: *App, device: vk.PhysicalDevice) !bool {
    // const properties = self.vk_instance.getPhysicalDeviceProperties(device);
    // const features = self.vk_instance.getPhysicalDeviceFeatures(device);

    const queue_families = try self.findQueueFamilies(device);
    const extensions_supported = try self.checkDeviceExtensionSupport(device);

    const swap_chain_adequate = blk: {
        if (!extensions_supported)
            break :blk false;

        const swap_chain_details = try self.querySwapChainSupport(device);
        defer allocator.free(swap_chain_details.formats);
        defer allocator.free(swap_chain_details.present_modes);
        break :blk swap_chain_details.formats.len != 0 and swap_chain_details.present_modes.len != 0;
    };

    const features = self.vk_instance.getPhysicalDeviceFeatures(device);

    return queue_families.isComplete() and extensions_supported and swap_chain_adequate and features.sampler_anisotropy == .true;
}

fn checkDeviceExtensionSupport(self: *App, device: vk.PhysicalDevice) !bool {
    const available_extensions = try self.vk_instance.enumerateDeviceExtensionPropertiesAlloc(device, null, allocator);
    defer allocator.free(available_extensions);

    outer: for (device_extensions) |required| {
        for (available_extensions) |available| {
            // Convert from type: [256]u8 to a null terminated slice [:0]const u8.
            // As the value of `available.extension_name` is always null terminated
            // according to the vulkan spec
            const name: [:0]const u8 = mem.span(@as([*:0]const u8, @ptrCast(&available.extension_name)));
            if (mem.eql(u8, name, mem.span(required))) {
                continue :outer;
            }
        }

        // required extension not found
        return false;
    }

    // All required extensions are found
    return true;
}

/// Creates the App.debug_messenger if debug mode is enabled
fn setupDebugMessenger(self: *App) !void {
    if (!enable_val_layers)
        return;

    const create_info = default_debug_messenger_create_info;
    self.debug_messenger = try self.vk_instance.createDebugUtilsMessengerEXT(&create_info, null);
}

fn mainLoop(self: *App) !void {
    start = std.time.milliTimestamp();
    while (!glfw.windowShouldClose(self.window)) {
        glfw.pollEvents();
        try self.drawFrame();
    }

    try self.vk_device.deviceWaitIdle();
}

fn drawFrame(self: *App) !void {
    _ = try self.vk_device.waitForFences(1, @ptrCast(&self.in_flight_fences[self.current_frame]), .true, std.math.maxInt(u64));

    // 'render_finished_semaphores' give validation errors because the semaphores might still be in use when rendering.
    // This isn't the case, but in more complex scenarios this could easily happen.
    // Therefore each image *should* have it's own semaphore for this.
    // But the tutorial doesn't do that so I'm not gonna bother....
    const image = self.vk_device.acquireNextImageKHR(self.swap_chain, std.math.maxInt(u64), self.image_available_semaphores[self.current_frame], .null_handle) catch |err| {
        // If the swapchain is out of date, recreate it. (for example when resizing)
        if (err == error.OutOfDateKHR) {
            try self.recreateSwapChain();
            return;
        }
        return err;
    };

    try self.updateUniformBuffer(self.current_frame);

    // Only reset the fences after we are sure we will do work
    // If fences are reset and we recreate swapchain and return early, then the program will deadlock.
    _ = try self.vk_device.resetFences(1, @ptrCast(&self.in_flight_fences[self.current_frame]));

    try self.vk_device.resetCommandBuffer(self.command_buffers[self.current_frame], .{});

    try self.recordCommandBuffer(self.command_buffers[self.current_frame], image.image_index);

    const submit_info: vk.SubmitInfo = .{
        .wait_semaphore_count = 1,
        .p_wait_semaphores = @ptrCast(&self.image_available_semaphores[self.current_frame]),
        .p_wait_dst_stage_mask = @ptrCast(&vk.PipelineStageFlags{ .color_attachment_output_bit = true }),
        .command_buffer_count = 1,
        .p_command_buffers = @ptrCast(&self.command_buffers[self.current_frame]),
        .signal_semaphore_count = 1,
        .p_signal_semaphores = @ptrCast(&self.render_finished_semaphores[self.current_frame]),
    };

    try self.graphics_queue.submit(1, @ptrCast(&submit_info), self.in_flight_fences[self.current_frame]);

    const present_info: vk.PresentInfoKHR = .{
        .wait_semaphore_count = 1,
        .p_wait_semaphores = @ptrCast(&self.render_finished_semaphores[self.current_frame]),
        .swapchain_count = 1,
        .p_swapchains = @ptrCast(&self.swap_chain),
        .p_image_indices = @ptrCast(&image.image_index),
    };

    const result = self.present_queue.presentKHR(&present_info) catch |err| blk: {
        if (err == error.OutOfDateKHR) {
            break :blk vk.Result.error_out_of_date_khr;
        } else {
            return err;
        }
    };

    if (result == .suboptimal_khr or result == .error_out_of_date_khr or self.framebuffer_resized) {
        self.framebuffer_resized = false;
        try self.recreateSwapChain();
    }

    self.current_frame = (self.current_frame + 1) % max_frames_in_flight;
}

var start: i64 = 0;
fn updateUniformBuffer(self: *App, current_image: u32) !void {
    const time = std.time.milliTimestamp() - start;
    var ubo: UniformBufferObject = .{
        .model = math.rotate(Mat4.identity, std.math.degreesToRadians(@as(f32, @floatFromInt(time))) / @as(f32, 1000) * 90, .{ .x = 0, .y = 0, .z = 1 }),
        .view = math.lookAt(.{ .x = 2, .y = 2, .z = 2 }, .{ .x = 0, .y = 0, .z = 0 }, .{ .x = 0, .y = 0, .z = 1 }),
        .proj = math.perspective(std.math.degreesToRadians(45), @as(f32, @floatFromInt(self.swap_chain_extent.width)) / @as(f32, @floatFromInt(self.swap_chain_extent.height)), 0.1, 10),
    };
    ubo.proj.y.y *= -1;

    const buffer: *UniformBufferObject = @ptrCast(@alignCast(self.uniform_buffers_mapped[current_image]));

    buffer.* = ubo;
}

fn cleanup(self: *App) void {
    try self.cleanupSwapChain();

    self.vk_device.destroySampler(self.texture_sampler, null);
    self.vk_device.destroyImageView(self.texture_image_view, null);
    self.vk_device.destroyImage(self.texture_image, null);
    self.vk_device.freeMemory(self.texture_image_memory, null);

    for (self.uniform_buffers, self.uniform_buffer_memories) |buffer, memory| {
        self.vk_device.destroyBuffer(buffer, null);
        self.vk_device.freeMemory(memory, null);
    }

    allocator.free(self.uniform_buffers);
    allocator.free(self.uniform_buffer_memories);
    allocator.free(self.uniform_buffers_mapped);

    self.vk_device.destroyDescriptorPool(self.descriptor_pool, null);
    self.vk_device.destroyDescriptorSetLayout(self.descriptor_set_layout, null);

    allocator.free(self.descriptor_sets);

    self.vk_device.destroyBuffer(self.index_buffer, null);
    self.vk_device.freeMemory(self.index_buffer_memory, null);

    self.vk_device.destroyBuffer(self.vertex_buffer, null);
    self.vk_device.freeMemory(self.vertex_buffer_memory, null);

    self.vk_device.destroyPipeline(self.graphics_pipeline, null);
    self.vk_device.destroyPipelineLayout(self.pipeline_layout, null);
    self.vk_device.destroyRenderPass(self.render_pass, null);

    inline for (0..max_frames_in_flight) |i| {
        self.vk_device.destroySemaphore(self.image_available_semaphores[i], null);
        self.vk_device.destroySemaphore(self.render_finished_semaphores[i], null);
        self.vk_device.destroyFence(self.in_flight_fences[i], null);
    }

    self.vk_device.destroyCommandPool(self.command_pool, null);

    self.vk_device.destroyDevice(null);

    if (enable_val_layers) {
        self.vk_instance.destroyDebugUtilsMessengerEXT(self.debug_messenger, null);
    }

    self.vk_instance.destroySurfaceKHR(self.surface, null);
    self.vk_instance.destroyInstance(null);
    glfw.destroyWindow(self.window);
    glfw.terminate();

    if (builtin.mode == .Debug) {
        const result = debug_allocator.deinit();
        if (result == .leak) {
            std.debug.print("\n\nPANIC PANIC EVERYBODY PANIC THERE IS A LEAK AAAAAAAAAAAAAAAAAAAA\n\n", .{});
        } else {
            std.debug.print("\n\nNo leaks detected\n\n", .{});
        }
    }
}

fn checkValLayerSupport(self: *App) !bool {
    const available_layers = try self.vk_base.enumerateInstanceLayerPropertiesAlloc(allocator);
    defer allocator.free(available_layers);

    outer: for (val_layers) |required| {
        for (available_layers) |available| {
            // Convert from type: [256]u8 to a null terminated slice [:0]const u8.
            // As the value of `available.layer_name` is always null terminated
            // according to the vulkan spec
            const name: [:0]const u8 = mem.span(@as([*:0]const u8, @ptrCast(&available.layer_name)));
            if (mem.eql(u8, mem.span(required), name)) {
                continue :outer;
            }
        }

        return false;
    }

    return true;
}

fn getRequiredExtensions() ![]const [*:0]const u8 {
    const glfw_extensions = try glfw.getRequiredInstanceExtensions();

    var extensions: std.ArrayList([*:0]const u8) = try .initCapacity(allocator, @intCast(glfw_extensions.len + 1));
    defer extensions.deinit(allocator);
    extensions.appendSliceAssumeCapacity(glfw_extensions[0..glfw_extensions.len]);

    if (enable_val_layers) {
        extensions.appendAssumeCapacity(vk.extensions.ext_debug_utils.name.ptr);
    }

    return try extensions.toOwnedSlice(allocator);
}

const QueueFamilyIndices = struct {
    graphics_family: ?u32 = null,
    present_family: ?u32 = null,

    pub fn isComplete(self: QueueFamilyIndices) bool {
        const fields = comptime std.meta.fieldNames(QueueFamilyIndices);
        inline for (fields) |field| {
            if (@field(self, field) == null) {
                return false;
            }
        }

        // If all fields have a value
        return true;
    }
};

fn findQueueFamilies(self: *const App, device: vk.PhysicalDevice) !QueueFamilyIndices {
    var indices: QueueFamilyIndices = .{};

    const queue_families = try self.vk_instance.getPhysicalDeviceQueueFamilyPropertiesAlloc(device, allocator);
    defer allocator.free(queue_families);

    for (queue_families, 0..) |queue_family, i| {
        const index: u32 = @intCast(i);

        if (queue_family.queue_flags.graphics_bit and indices.graphics_family == null) {
            indices.graphics_family = index;
        }

        if (try self.vk_instance.getPhysicalDeviceSurfaceSupportKHR(device, index, self.surface) == .true) {
            indices.present_family = index;
        }

        if (indices.isComplete()) {
            return indices;
        }
    }

    return indices;
}

const SwapChainSupportDetails = struct {
    capabilites: vk.SurfaceCapabilitiesKHR,
    formats: []vk.SurfaceFormatKHR,
    present_modes: []vk.PresentModeKHR,
};

fn querySwapChainSupport(self: *const App, device: vk.PhysicalDevice) !SwapChainSupportDetails {
    const capabilities = try self.vk_instance.getPhysicalDeviceSurfaceCapabilitiesKHR(device, self.surface);
    const formats = try self.vk_instance.getPhysicalDeviceSurfaceFormatsAllocKHR(device, self.surface, allocator);
    const present_modes = try self.vk_instance.getPhysicalDeviceSurfacePresentModesAllocKHR(device, self.surface, allocator);

    return SwapChainSupportDetails{
        .capabilites = capabilities,
        .formats = formats,
        .present_modes = present_modes,
    };
}

fn chooseSwapSurfaceFormat(_: App, formats: []const vk.SurfaceFormatKHR) vk.SurfaceFormatKHR {
    for (formats) |format| {
        if (format.format == .b8g8r8a8_srgb and format.color_space == .srgb_nonlinear_khr) {
            return format;
        }
    }

    return formats[0];
}

fn chooseSwapPresentMode(_: App, present_modes: []const vk.PresentModeKHR) vk.PresentModeKHR {
    for (present_modes) |present_mode| {
        // Tripple buffering vsync
        if (present_mode == .mailbox_khr)
            return present_mode;
    }

    // Vsync
    return vk.PresentModeKHR.fifo_khr;
}

fn chooseSwapExtent(self: *App, capabilities: *const vk.SurfaceCapabilitiesKHR) vk.Extent2D {
    if (capabilities.current_extent.width != std.math.maxInt(u32)) {
        return capabilities.current_extent;
    }

    var framebuffer_width: c_int, var framebuffer_height: c_int = .{ undefined, undefined };
    glfw.getFramebufferSize(self.window, &framebuffer_width, &framebuffer_height);

    return vk.Extent2D{
        .width = std.math.clamp(@as(u32, @intCast(framebuffer_width)), capabilities.min_image_extent.width, capabilities.max_image_extent.width),
        .height = std.math.clamp(@as(u32, @intCast(framebuffer_height)), capabilities.min_image_extent.height, capabilities.max_image_extent.height),
    };
}

fn debugCallback(
    message_severity: vk.DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk.DebugUtilsMessageTypeFlagsEXT,
    callback_data: ?*const vk.DebugUtilsMessengerCallbackDataEXT,
    user_data: ?*anyopaque,
) callconv(vk.vulkan_call_conv) vk.Bool32 {
    _ = user_data;
    const severity = if (message_severity.info_bit_ext) "INFO" else if (message_severity.verbose_bit_ext) "VERBOSE" else if (message_severity.warning_bit_ext) "\x1b[33mWARNING\x1b[0m" else if (message_severity.error_bit_ext) "\x1b[31mERROR\x1b[0m" else "UNKNOWN";
    const @"type" = if (message_type.general_bit_ext) "GENERAL" else if (message_type.validation_bit_ext) "VALIDATION" else if (message_type.performance_bit_ext) "\x1b[34mPERFORMANCE\x1b[0m" else if (message_type.device_address_binding_bit_ext) "DEVICE ADDRESS BINDING" else "UNKNOWN";

    std.debug.print("\n[DEBUG] [{s}] [{s}]: {s}\n", .{ severity, @"type", callback_data.?.p_message.? });

    return .false;
}

const default_debug_messenger_create_info: vk.DebugUtilsMessengerCreateInfoEXT = .{
    // Enable all kinds of Validation messages
    .message_severity = .{
        .error_bit_ext = true,
        .warning_bit_ext = true,
        .verbose_bit_ext = true,
        // Info is WAAAY too verbose
        .info_bit_ext = false,
    },
    // Enabled all kinds of Validation types
    .message_type = .{
        .performance_bit_ext = true,
        .validation_bit_ext = true,
        .general_bit_ext = true,
        // Requires extension, not in core
        .device_address_binding_bit_ext = false,
    },
    .pfn_user_callback = debugCallback,
};

// Define the function ourselves because glfw.getInstanceProcAddr doesn't use correct types
extern fn glfwGetInstanceProcAddress(vk.Instance, [*:0]const u8) callconv(.c) vk.PfnVoidFunction;
