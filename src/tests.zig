const std = @import("std");
const mott = @import("main.zig");

comptime {
    _ = @import("tests/syntax.zig");
    _ = @import("tests/behaviour.zig");
}
