const std = @import("std");
const makeshift = @import("main.zig");

comptime {
    _ = @import("tests/syntax.zig");
    _ = @import("tests/behaviour.zig");
}
