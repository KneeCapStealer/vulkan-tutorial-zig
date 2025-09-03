const std = @import("std");
const assert = std.debug.assert;

pub const Vec2 = extern struct {
    x: f32,
    y: f32,
};

pub const Vec3 = extern struct {
    x: f32,
    y: f32,
    z: f32,

    pub fn normalize(vec: Vec3) Vec3 {
        const len = vec.lenght();

        return Vec3{ .x = vec.x / len, .y = vec.y / len, .z = vec.z / len };
    }

    pub fn lenght(vec: Vec3) f32 {
        return std.math.sqrt(
            (std.math.pow(f32, vec.x, 2) + std.math.pow(f32, vec.y, 2) + std.math.pow(f32, vec.z, 2)),
        );
    }

    pub fn cross(a: Vec3, b: Vec3) Vec3 {
        return Vec3{
            .x = a.y * b.z - a.z * b.y,
            .y = a.z * b.x - a.x * b.z,
            .z = a.x * b.x - a.y * b.x,
        };
    }

    pub fn multiplyBy(vec: Vec3, a: f32) Vec3 {
        return Vec3{
            .x = vec.x * a,
            .y = vec.y * a,
            .z = vec.z * a,
        };
    }

    pub fn diff(lhs: Vec3, rhs: Vec3) Vec3 {
        return Vec3{
            .x = lhs.x - rhs.x,
            .y = lhs.y - rhs.y,
            .z = lhs.z - rhs.z,
        };
    }

    pub fn dot(a: Vec3, b: Vec3) f32 {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }
};

pub const Vec4 = extern struct {
    x: f32,
    y: f32,
    z: f32,
    w: f32,

    pub const zero: Vec4 = .{ .x = 0, .y = 0, .z = 0, .w = 0 };

    pub fn multiplyBy(vec: Vec4, a: f32) Vec4 {
        return Vec4{
            .x = vec.x * a,
            .y = vec.y * a,
            .z = vec.z * a,
            .w = vec.w * a,
        };
    }

    pub fn sum(lhs: Vec4, rhs: Vec4) Vec4 {
        return Vec4{
            .x = lhs.x + rhs.x,
            .y = lhs.y + rhs.y,
            .z = lhs.z + rhs.z,
            .w = lhs.w + rhs.w,
        };
    }

    pub fn dot(a: Vec4, b: Vec4) f32 {
        return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
};

pub const Mat4 = extern struct {
    x: Vec4,
    y: Vec4,
    z: Vec4,
    w: Vec4,

    pub const zero: Mat4 = .{
        .x = .zero,
        .y = .zero,
        .z = .zero,
        .w = .zero,
    };

    pub const identity: Mat4 = .{
        .x = .{ .x = 1, .y = 0, .z = 0, .w = 0 },
        .y = .{ .x = 0, .y = 1, .z = 0, .w = 0 },
        .z = .{ .x = 0, .y = 0, .z = 1, .w = 0 },
        .w = .{ .x = 0, .y = 0, .z = 0, .w = 1 },
    };
};

/// Stolen from glm
pub fn rotate(m: Mat4, angle: f32, v: Vec3) Mat4 {
    const a: f32 = angle;
    const c = @cos(a);
    const s = @sin(a);

    const axis = v.normalize();
    const temp = axis.multiplyBy(1 - c);

    const rotate_matrix: Mat4 = .{
        .x = .{
            .x = c + temp.x * axis.x,
            .y = temp.x * axis.y + s * axis.z,
            .z = temp.x * axis.z - s * axis.y,
            .w = 0,
        },
        .y = .{
            .x = temp.y * axis.x - s * axis.z,
            .y = c + temp.y * axis.y,
            .z = temp.y * axis.z + s * axis.x,
            .w = 0,
        },
        .z = .{
            .x = temp.z * axis.x + s * axis.y,
            .y = temp.z * axis.y - s * axis.x,
            .z = c + temp.z * axis.z,
            .w = 0,
        },
        .w = .{ .x = 0, .y = 0, .z = 0, .w = 0 },
    };

    return Mat4{
        .x = Vec4.sum(Vec4.sum(m.x.multiplyBy(rotate_matrix.x.x), m.y.multiplyBy(rotate_matrix.x.y)), m.z.multiplyBy(rotate_matrix.x.z)),
        .y = Vec4.sum(Vec4.sum(m.x.multiplyBy(rotate_matrix.y.x), m.y.multiplyBy(rotate_matrix.y.y)), m.z.multiplyBy(rotate_matrix.y.z)),
        .z = Vec4.sum(Vec4.sum(m.x.multiplyBy(rotate_matrix.z.x), m.y.multiplyBy(rotate_matrix.z.y)), m.z.multiplyBy(rotate_matrix.z.z)),
        .w = m.w,
    };
}

// glm lookAtLH
pub fn lookAt(eye: Vec3, center: Vec3, up: Vec3) Mat4 {
    const f: Vec3 = .normalize(Vec3.diff(center, eye));
    const s: Vec3 = .normalize(Vec3.cross(up, f));
    const u: Vec3 = .cross(f, s);

    var result: Mat4 = .identity;
    result.x.x = s.x;
    result.y.x = s.y;
    result.z.x = s.z;
    result.x.y = u.x;
    result.y.y = u.y;
    result.z.y = u.z;
    result.x.z = f.x;
    result.y.z = f.y;
    result.z.z = f.z;
    result.w.x = -Vec3.dot(s, eye);
    result.w.y = -Vec3.dot(u, eye);
    result.w.z = -Vec3.dot(f, eye);

    return result;
}

// use glm perspectiveLH_ZO which means perspective Left Handed Zero One
// As in left handed coordinate system with a 0 to 1 depth range.
pub fn perspective(fovy: f32, aspect: f32, z_near: f32, z_far: f32) Mat4 {
    const abs_aspect = if (aspect > 0) aspect else -aspect;
    assert((abs_aspect - std.math.floatEps(f32)) > @as(f32, 0));

    const tanHalfFovy: f32 = @tan(fovy / @as(f32, 2));

    var result: Mat4 = .zero;
    result.x.x = @as(f32, 1) / (aspect * tanHalfFovy);
    result.y.y = @as(f32, 1) / tanHalfFovy;
    result.z.z = z_far / (z_far - z_near);
    result.z.w = @as(f32, 1);
    result.w.w = -(z_far * z_near) / (z_far - z_near);

    return result;
}
