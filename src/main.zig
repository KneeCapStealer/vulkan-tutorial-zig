const App = @import("App.zig");

pub fn main() !void {
    var app = App.init();

    try app.run();
}
