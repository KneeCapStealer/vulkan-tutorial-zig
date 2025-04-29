const std = @import("std");
const glfw = @import("glfw");
const vk = @import("vulkan");

const App = @This();

const width = 800;
const height = 600;

window: *glfw.Window,

pub fn init() App {
    return .{
        .window = undefined,
    };
}

pub fn run(self: *App) !void {
    try self.initVulkan();
    try self.mainLoop();
    try self.cleanup();
}

fn initVulkan(self: *App) !void {
    try glfw.init();
    glfw.windowHint(glfw.ClientAPI, glfw.NoAPI);
    glfw.windowHint(glfw.Resizable, 0);

    self.window = try glfw.createWindow(width, height, "Vulkan", null, null);
}

fn mainLoop(self: *App) !void {
    while (!glfw.windowShouldClose(self.window)) {
        glfw.pollEvents();
    }
}

fn cleanup(self: *App) !void {
    glfw.destroyWindow(self.window);
    glfw.terminate();
}
