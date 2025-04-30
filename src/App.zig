const std = @import("std");
const glfw = @import("glfw");
const vk = @import("vulkan");

const App = @This();

const allocator = if (@import("builtin").mode == .Debug) (std.heap.DebugAllocator(.{}){}).allocator() else std.heap.smp_allocator;

const width = 800;
const height = 600;

window: *glfw.Window,
vk_base: vk.BaseWrapper,
instance_wrapper: vk.InstanceWrapper,
vk_instance: vk.InstanceProxy,

pub fn init() App {
    return .{
        .window = undefined,
        .vk_base = undefined,
        .instance_wrapper = undefined,
        .vk_instance = undefined,
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
    const appInfo: vk.ApplicationInfo = .{
        .p_application_name = "Hello Triangle",
        .p_engine_name = "No Engine",
        .engine_version = @bitCast(vk.makeApiVersion(1, 0, 0, 0)),
        .application_version = @bitCast(vk.makeApiVersion(1, 0, 0, 0)),
        .api_version = @bitCast(vk.API_VERSION_1_0),
    };

    var glfw_extension_count: u32 = undefined;
    const glfw_extension = glfw.getRequiredInstanceExtensions(&glfw_extension_count);

    const instance_create_info: vk.InstanceCreateInfo = .{
        .p_application_info = &appInfo,
        .enabled_layer_count = 0,
        .enabled_extension_count = glfw_extension_count,
        .pp_enabled_extension_names = glfw_extension,
    };
    const instance = try self.vk_base.createInstance(&instance_create_info, null);
    self.instance_wrapper = .load(instance, self.vk_base.dispatch.vkGetInstanceProcAddr.?);
    self.vk_instance = .init(instance, &self.instance_wrapper);
}

fn initVulkan(self: *App) !void {
    self.vk_base = .load(glfwGetInstanceProcAddress);

    try self.createInstance();
}

fn mainLoop(self: *App) !void {
    while (!glfw.windowShouldClose(self.window)) {
        glfw.pollEvents();
    }
}

fn cleanup(self: *App) !void {
    self.vk_instance.destroyInstance(null);
    glfw.destroyWindow(self.window);
    glfw.terminate();
}

// Define the function ourselves because glfw.getInstanceProcAddr doesn't use correct types
extern fn glfwGetInstanceProcAddress(vk.Instance, [*:0]const u8) callconv(.C) vk.PfnVoidFunction;
