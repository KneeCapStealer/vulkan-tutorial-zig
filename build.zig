const std = @import("std");

// Although this function looks imperative, note that its job is to
// declaratively construct a build graph that will be executed by an external
// runner.
pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const exe = b.addExecutable(.{
        .name = "waltuh",
        .root_module = exe_mod,
    });
    b.installArtifact(exe);
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
    exe.root_module.addImport("vulkan", vulkan);

    const glfw = b.dependency("zglfw", .{
        .target = target,
        .optimize = optimize,
        .x11 = true,
        .wayland = true,
    });
    exe.root_module.addImport("glfw", glfw.module("root"));

    if (target.result.os.tag != .emscripten) {
        exe.linkLibrary(glfw.artifact("glfw"));
    }

    const zigimg_dependency = b.dependency("zigimg", .{
        .target = target,
        .optimize = optimize,
    });

    exe.root_module.addImport("zigimg", zigimg_dependency.module("zigimg"));

    const obj_mod = b.dependency("obj", .{ .target = target, .optimize = optimize }).module("obj");
    exe_mod.addImport("obj", obj_mod);

    const libwindow = b.dependency("master", .{
        .target = target,
        .optimize = optimize,
        .registry = b.path("vulkan/vk.xml"),
    });
    exe_mod.addImport("libwindow", libwindow.module("renderer"));

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
