const std = @import("std");
const mem = std.mem;
const builtin = @import("builtin");

const glfw = @import("glfw");
const vk = @import("vulkan");

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

window: *glfw.Window,
vk_base: vk.BaseWrapper,
instance_wrapper: vk.InstanceWrapper,
vk_instance: vk.InstanceProxy,
vk_physical: vk.PhysicalDevice,
device_wrapper: vk.DeviceWrapper,
vk_device: vk.DeviceProxy,
graphics_queue: vk.QueueProxy,
present_queue: vk.QueueProxy,
debug_messenger: vk.DebugUtilsMessengerEXT,
surface: vk.SurfaceKHR,

pub fn init() App {
    return App{
        .window = undefined,
        .vk_base = undefined,
        .instance_wrapper = undefined,
        .vk_instance = undefined,
        .vk_physical = .null_handle,
        .device_wrapper = undefined,
        .vk_device = undefined,
        .graphics_queue = undefined,
        .present_queue = undefined,
        .debug_messenger = .null_handle,
        .surface = .null_handle,
    };
}

pub fn run(self: *App) !void {
    try glfw.init();

    try self.initWindow();
    try self.initVulkan();
    try self.mainLoop();
    try self.cleanup();
}

fn initWindow(self: *App) !void {
    glfw.windowHint(glfw.ClientAPI, glfw.NoAPI);
    glfw.windowHint(glfw.Resizable, 0);

    self.window = try glfw.createWindow(width, height, "Vulkan", null, null);
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
}

/// Create logical device to encapsulate physical device
fn createLogicalDevice(self: *App) !void {
    const indices = try self.findQueueFamilies(self.vk_physical);
    if (!indices.isComplete()) {
        return error.FailedToFindQueueFamilies;
    }

    var unique_queue_families: @import("set").Set(u32) = try .initCapacity(allocator, 1);
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
    };

    // Device creation
    const device = try self.vk_instance.createDevice(self.vk_physical, &create_info, null);
    self.device_wrapper = .load(device, self.instance_wrapper.dispatch.vkGetDeviceProcAddr.?);
    self.vk_device = .init(device, &self.device_wrapper);

    // Device Queue handles
    const graphics_queue = self.vk_device.getDeviceQueue(indices.graphics_family.?, 0);
    self.graphics_queue = .init(graphics_queue, &self.device_wrapper);

    const present_queue = self.vk_device.getDeviceQueue(indices.present_family.?, 0);
    self.graphics_queue = .init(present_queue, &self.device_wrapper);
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

    return is_discrete and queue_families.isComplete();
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
        try self.cleanup();
        std.process.exit(0);
    }
}

fn cleanup(self: *App) !void {
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

    outer: for (val_layers) |layer_name| {
        const layer_name_len = mem.len(layer_name);

        for (available_layers) |layer| {
            if (mem.eql(u8, mem.span(layer_name), layer.layer_name[0..layer_name_len])) {
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
