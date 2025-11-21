const std = @import("std");

// Although this function looks imperative, note that its job is to
// declaratively construct a build graph that will be executed by an external
// runner.
pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    b.installDirectory(.{
        .source_dir = b.path("images"),
        .install_dir = .prefix,
        .install_subdir = "images",
    });
    b.installDirectory(.{
        .source_dir = b.path("shaders/out"),
        .install_dir = .prefix,
        .install_subdir = "shaders/out",
    });
    b.installDirectory(.{
        .source_dir = b.path("objects"),
        .install_dir = .prefix,
        .install_subdir = "objects",
    });

    const vulkan = b.dependency("vulkan", .{
        .registry = b.path("vulkan/vk.xml"),
        .target = target,
        .optimize = optimize,
    }).module("vulkan-zig");

    const zigimg = b.dependency("zigimg", .{
        .target = target,
        .optimize = optimize,
    }).module("zigimg");

    const obj = b.dependency("obj", .{
        .target = target,
        .optimize = optimize,
    }).module("obj");

    const libwindow = b.dependency("libwindow", .{
        .target = target,
        .optimize = optimize,
    }).module("libwindow");

    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "vulkan", .module = vulkan },
            .{ .name = "zigimg", .module = zigimg },
            .{ .name = "obj", .module = obj },
            .{ .name = "libwindow", .module = libwindow },
        },
    });

    const exe = b.addExecutable(.{
        .name = "waltuh",
        .root_module = exe_mod,
    });

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    // This allows the user to pass arguments to the application in the build
    // command itself, like this: `zig build run -- arg1 arg2 etc`
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    const exe_unit_tests = b.addTest(.{
        .root_module = exe_mod,
    });

    const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);

    // Similar to creating the run step earlier, this exposes a `test` step to
    // the `zig build --help` menu, providing a way for the user to request
    // running the unit tests.
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_exe_unit_tests.step);
}
