const std = @import("std");
const vk = @import("vulkan");

pub fn print() void {
    std.debug.print("Hello, World!", .{});
}

pub fn main() !void {
    var once = std.once(print);

    once.call();
    once.call();
    once.call();
    once.call();
    once.call();
    once.call();
    once.call();
    once.call();
}
