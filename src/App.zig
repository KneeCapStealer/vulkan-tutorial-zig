const std = @import("std");
const mem = std.mem;
const builtin = @import("builtin");

const glfw = @import("glfw");
const vk = @import("vulkan");
const set = @import("set");

const math = @import("math.zig");
const Vec2 = math.vec2;
const Vec3 = math.vec3;

const Vertex = struct {
    pos: Vec2,
    color: Vec3,

    fn getBindingDescription() vk.VertexInputBindingDescription {
        const binding_description: vk.VertexInputBindingDescription = .{
            .binding = 0,
            .stride = @sizeOf(@This()),
            .input_rate = .vertex,
        };

        return binding_description;
    }

    fn getAttributeDescriptions() [2]vk.VertexInputAttributeDescription {
        return [2]vk.VertexInputAttributeDescription{ .{
            .binding = 0,
            .location = 0,
            .format = .r32g32_sfloat,
            .offset = @offsetOf(@This(), "pos"),
        }, .{
            .binding = 0,
            .location = 1,
            .format = .r32g32b32_sfloat,
            .offset = @offsetOf(@This(), "color"),
        } };
    }
};

const App = @This();

var debugAllocator = std.heap.DebugAllocator(.{}){};
const allocator = if (builtin.mode == .Debug) debugAllocator.allocator() else std.heap.smp_allocator;

const width = 800;
const height = 600;

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
pipeline_layout: vk.PipelineLayout,
graphics_pipeline: vk.Pipeline,

command_pool: vk.CommandPool,
command_buffers: [max_frames_in_flight]vk.CommandBuffer,

image_available_semaphores: [max_frames_in_flight]vk.Semaphore,
render_finished_semaphores: [max_frames_in_flight]vk.Semaphore,
in_flight_fences: [max_frames_in_flight]vk.Fence,

current_frame: u32,

framebuffer_resized: bool,

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
        .pipeline_layout = .null_handle,
        .graphics_pipeline = .null_handle,

        .command_pool = .null_handle,
        .command_buffers = command_buffers,

        .image_available_semaphores = image_available_semaphores,
        .render_finished_semaphores = render_finished_semaphores,
        .in_flight_fences = in_flight_fences,

        .current_frame = 0,

        .framebuffer_resized = false,
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
    glfw.windowHint(glfw.ClientAPI, glfw.NoAPI);

    self.window = try glfw.createWindow(width, height, "Vulkan", null, null);
    glfw.setWindowUserPointer(self.window, self);
    _ = glfw.setFramebufferSizeCallback(self.window, framebufferResizeCallback);
}

fn framebufferResizeCallback(window: *glfw.Window, _: c_int, _: c_int) callconv(.C) void {
    const self: *App = @alignCast(@ptrCast(glfw.getWindowUserPointer(window)));

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
    if (glfw.createWindowSurface(@intFromEnum(self.vk_instance.handle), self.window, null, @ptrCast(&self.surface)) != glfw.VkResult.success) {
        return error.WindowSurfaceCreationFailed;
    }
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
    try self.createGraphicsPipeline();
    try self.createFramebuffers();
    try self.createCommandPool();
    try self.createCommandBuffers();
    try self.createSyncObjects();
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

    const clear_color: vk.ClearValue = .{ .color = .{ .float_32 = .{ 0, 0, 0, 1 } } };
    const render_pass_info: vk.RenderPassBeginInfo = .{
        .render_pass = self.render_pass,
        .framebuffer = self.swap_chain_framebuffers[image_index],
        .render_area = .{ .extent = self.swap_chain_extent, .offset = .{ .x = 0, .y = 0 } },
        .clear_value_count = 1,
        .p_clear_values = @ptrCast(&clear_color),
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

    cmd_buf.draw(3, 1, 0, 0);
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
        const attachments: [1]vk.ImageView = .{image_view};

        const framebuffer_info: vk.FramebufferCreateInfo = .{
            .render_pass = self.render_pass,
            // Specifies the attachments bound to the renderpass.
            // When the attachment at position 0 is written to, the image view at position 0 will be used
            .attachment_count = 1,
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

    const color_Attachment_ref: vk.AttachmentReference = .{ .attachment = 0, .layout = .color_attachment_optimal };

    const subpass: vk.SubpassDescription = .{
        .pipeline_bind_point = .graphics,
        .color_attachment_count = 1,
        .p_color_attachments = @ptrCast(&color_Attachment_ref),
    };

    // Wait for the color attachment output stage is available for write
    const dependency: vk.SubpassDependency = .{
        .src_subpass = vk.SUBPASS_EXTERNAL,
        .dst_subpass = 0,
        .src_stage_mask = .{ .color_attachment_output_bit = true },
        .src_access_mask = .{},
        .dst_stage_mask = .{ .color_attachment_output_bit = true },
        .dst_access_mask = .{ .color_attachment_write_bit = true },
    };

    const render_pass_info: vk.RenderPassCreateInfo = .{
        .attachment_count = 1,
        .p_attachments = @ptrCast(&color_attachment),
        .subpass_count = 1,
        .p_subpasses = @ptrCast(&subpass),
        .dependency_count = 1,
        .p_dependencies = @ptrCast(&dependency),
    };

    self.render_pass = try self.vk_device.createRenderPass(&render_pass_info, null);
}

fn createGraphicsPipeline(self: *App) !void {
    const vertex_shader = try std.fs.cwd().openFile("shaders/out/vert.spv", std.fs.File.OpenFlags{
        .lock = .shared,
        .mode = .read_only,
    });

    errdefer vertex_shader.close();
    const fragment_shader = try std.fs.cwd().openFile("shaders/out/frag.spv", std.fs.File.OpenFlags{
        .lock = .shared,
        .mode = .read_only,
    });
    errdefer fragment_shader.close();

    // Shader creation
    const vert_code = try vertex_shader.readToEndAlloc(allocator, std.math.maxInt(usize));
    vertex_shader.close();
    const frag_code = try fragment_shader.readToEndAlloc(allocator, std.math.maxInt(usize));
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
    const input_assembly: vk.PipelineInputAssemblyStateCreateInfo = .{ .topology = .triangle_list, .primitive_restart_enable = vk.FALSE };

    const viewport: vk.Viewport = .{
        .width = @floatFromInt(self.swap_chain_extent.width),
        .height = @floatFromInt(self.swap_chain_extent.height),
        .x = 0,
        .y = 0,
        .max_depth = 1,
        .min_depth = 0,
    };

    const scissor: vk.Rect2D = .{ .extent = self.swap_chain_extent, .offset = .{ .y = 0, .x = 0 } };

    const viewport_state: vk.PipelineViewportStateCreateInfo = .{ .viewport_count = 1, .p_viewports = @ptrCast(&viewport), .scissor_count = 1, .p_scissors = @ptrCast(&scissor) };

    const rasterizer: vk.PipelineRasterizationStateCreateInfo = .{
        .depth_clamp_enable = vk.FALSE,
        .rasterizer_discard_enable = vk.FALSE,
        .polygon_mode = .fill,
        .line_width = 1,
        .cull_mode = .{ .back_bit = true },
        .front_face = .clockwise,
        .depth_bias_enable = vk.FALSE,
        .depth_bias_clamp = 0,
        .depth_bias_constant_factor = 0,
        .depth_bias_slope_factor = 0,
    };

    const multisampling: vk.PipelineMultisampleStateCreateInfo = .{
        .sample_shading_enable = vk.FALSE,
        .rasterization_samples = .{ .@"1_bit" = true },
        .min_sample_shading = 1,
        .alpha_to_coverage_enable = vk.FALSE,
        .alpha_to_one_enable = vk.FALSE,
    };

    const color_blend_attachment: vk.PipelineColorBlendAttachmentState = .{
        // Use all colors
        .color_write_mask = .{ .a_bit = true, .b_bit = true, .g_bit = true, .r_bit = true },

        // Don't blend colors with previos colors
        .blend_enable = vk.FALSE,
        .src_color_blend_factor = .one,
        .dst_color_blend_factor = .zero,
        .color_blend_op = .add,
        .src_alpha_blend_factor = .one,
        .dst_alpha_blend_factor = .zero,
        .alpha_blend_op = .add,
    };

    const color_blend: vk.PipelineColorBlendStateCreateInfo = .{
        // Again don't do color blending
        .logic_op_enable = vk.FALSE,
        .logic_op = .copy,
        .attachment_count = 1,
        .p_attachments = @ptrCast(&color_blend_attachment),
        .blend_constants = .{ 0.0, 0.0, 0.0, 0.0 },
    };

    const pipeline_layout_info: vk.PipelineLayoutCreateInfo = .{};

    self.pipeline_layout = try self.vk_device.createPipelineLayout(&pipeline_layout_info, null);

    const pipeline_info: vk.GraphicsPipelineCreateInfo = .{
        // Vertex and fragment shaders
        .stage_count = 2,
        .p_stages = &shader_stages,
        .p_vertex_input_state = &vertex_input_info,
        .p_input_assembly_state = &input_assembly,
        .p_viewport_state = &viewport_state,
        .p_rasterization_state = &rasterizer,
        .p_multisample_state = &multisampling,
        .p_depth_stencil_state = null,
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
        const create_info: vk.ImageViewCreateInfo = .{
            .image = image,
            .view_type = .@"2d",
            .format = self.swap_chain_image_format,
            .components = .{
                .a = .identity,
                .b = .identity,
                .g = .identity,
                .r = .identity,
            },
            .subresource_range = .{
                .aspect_mask = .{ .color_bit = true },
                .base_mip_level = 0,
                .layer_count = 1,
                .level_count = 1,
                .base_array_layer = 0,
            },
        };

        self.swap_chain_image_views[i] = try self.vk_device.createImageView(&create_info, null);
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
    try self.createFramebuffers();
}

fn cleanupSwapChain(self: *App) !void {
    for (self.swap_chain_framebuffers) |framebuffer| {
        self.vk_device.destroyFramebuffer(framebuffer, null);
    }

    for (self.swap_chain_image_views) |view| {
        self.vk_device.destroyImageView(view, null);
    }

    self.vk_device.destroySwapchainKHR(self.swap_chain, null);
}

fn createSwapChain(self: *App) !void {
    const swap_chain_details = try self.querySwapChainSupport(self.vk_physical);

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
        .clipped = vk.TRUE,
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

    var unique_queue_families: set.Set(u32) = try .initCapacity(allocator, 1);
    _ = try unique_queue_families.appendSlice(&.{ indices.present_family.?, indices.graphics_family.? });

    var iter = unique_queue_families.iterator();
    var queue_create_infos: std.ArrayList(vk.DeviceQueueCreateInfo) = try .initCapacity(allocator, 1);
    defer queue_create_infos.deinit();

    const queue_prio: f32 = 1.0;

    var i: usize = 0;
    while (iter.next()) |queue_family| : (i += 1) {
        try queue_create_infos.append(vk.DeviceQueueCreateInfo{
            .queue_family_index = queue_family.*,
            .queue_count = 1,
            .p_queue_priorities = @ptrCast(&queue_prio),
        });
    }

    // Activate no features for now
    const device_features: vk.PhysicalDeviceFeatures = .{};
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
    const properties = self.vk_instance.getPhysicalDeviceProperties(device);
    // const features = self.vk_instance.getPhysicalDeviceFeatures(device);

    const is_discrete = properties.device_type == .discrete_gpu;
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

    return is_discrete and queue_families.isComplete() and extensions_supported and swap_chain_adequate;
}

fn checkDeviceExtensionSupport(self: *App, device: vk.PhysicalDevice) !bool {
    const available_extensions = try self.vk_instance.enumerateDeviceExtensionPropertiesAlloc(device, null, allocator);

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
    while (!glfw.windowShouldClose(self.window)) {
        glfw.pollEvents();
        try self.drawFrame();
    }

    try self.vk_device.deviceWaitIdle();
}

fn drawFrame(self: *App) !void {
    _ = try self.vk_device.waitForFences(1, @ptrCast(&self.in_flight_fences[self.current_frame]), vk.TRUE, std.math.maxInt(u64));

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

fn cleanup(self: *App) void {
    try self.cleanupSwapChain();

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
}

fn checkValLayerSupport(self: *App) !bool {
    const available_layers = try self.vk_base.enumerateInstanceLayerPropertiesAlloc(allocator);

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
    var glfw_extension_count: u32 = undefined;
    const glfw_extensions = glfw.getRequiredInstanceExtensions(&glfw_extension_count) orelse (&[_][*:0]const u8{}).ptr;

    var extensions: std.ArrayList([*:0]const u8) = try .initCapacity(allocator, @intCast(glfw_extension_count + 1));
    extensions.appendSliceAssumeCapacity(glfw_extensions[0..glfw_extension_count]);

    if (enable_val_layers) {
        extensions.appendAssumeCapacity(vk.extensions.ext_debug_utils.name.ptr);
    }

    return try extensions.toOwnedSlice();
}

const QueueFamilyIndices = struct {
    graphics_family: ?u32 = null,
    present_family: ?u32 = null,

    pub fn isComplete(self: QueueFamilyIndices) bool {
        return self.graphics_family != null and self.present_family != null;
    }
};

fn findQueueFamilies(self: *App, device: vk.PhysicalDevice) !QueueFamilyIndices {
    var indices: QueueFamilyIndices = .{};

    const queue_families = try self.vk_instance.getPhysicalDeviceQueueFamilyPropertiesAlloc(device, allocator);
    for (queue_families, 0..) |queue_family, i| {
        const index: u32 = @intCast(i);

        if (queue_family.queue_flags.graphics_bit) {
            indices.graphics_family = index;
        }

        if (try self.vk_instance.getPhysicalDeviceSurfaceSupportKHR(device, index, self.surface) != 0) {
            indices.present_family = index;
        }

        if (indices.isComplete()) {
            break;
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

fn chooseSwapSurfaceFormat(_: *const App, formats: []const vk.SurfaceFormatKHR) vk.SurfaceFormatKHR {
    for (formats) |format| {
        if (format.format == .b8g8r8a8_srgb and format.color_space == .srgb_nonlinear_khr) {
            return format;
        }
    }

    return formats[0];
}

fn chooseSwapPresentMode(_: *const App, present_modes: []const vk.PresentModeKHR) vk.PresentModeKHR {
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

    return vk.FALSE;
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
extern fn glfwGetInstanceProcAddress(vk.Instance, [*:0]const u8) callconv(.C) vk.PfnVoidFunction;
