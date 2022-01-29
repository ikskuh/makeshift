const std = @import("std");
const ihex = @import("ihex");
const ptk = @import("parser-toolkit");

pub fn main() anyerror!void {
    const demo_source = @embedFile("../docs/concept.mott");

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var ast = try Parser.parse(gpa.allocator(), demo_source);
    defer ast.deinit();

    const stdout = std.io.getStdOut();
    try AstPrinter(std.fs.File.Writer).print(ast, stdout.writer());
}

const Environment = struct {
    memory: [32768]u16,
};

const Interpreter = struct {};

const Parser = struct {
    const Self = @This();

    fn parse(allocator: std.mem.Allocator, source: []const u8) !Ast {
        var tokenizer = Tokenizer.init(source);
        var core = ParserCore.init(&tokenizer);

        var ast = Ast.init(allocator);
        errdefer ast.deinit();

        while (try core.nextToken()) |token| {
            std.debug.print("{s} <= '{s}'\n", .{ token.type, token.text });
        }

        return ast;
    }
};

fn AstPrinter(comptime Writer: type) type {
    return struct {
        pub const Error = Writer.Error;

        const indent_char = "\t";
        const indent_level = 2;

        fn print(ast: Ast, writer: Writer) Error!void {
            var next_node = ast.top_level;
            while (next_node) |node| : (next_node = node.next) {
                switch (node.data) {
                    .@"var" => |val| try printVarDecl(val, writer),
                    .@"const" => |val| try printConstDecl(val, writer),
                    .@"fn" => |val| try printFnDecl(val, writer),
                }
            }
        }

        fn printVarDecl(decl: Ast.VarDeclaration, writer: Writer) Error!void {
            if (decl.value) |value| {
                try writer.print("var {s} = ", .{decl.name});
                try printExpr(value, writer);
                try writer.writeAll(";\n");
            } else {
                try writer.print("var {s};\n", .{decl.name});
            }
        }

        fn printConstDecl(decl: Ast.ConstDeclaration, writer: Writer) Error!void {
            try writer.print("const {s} = ", .{decl.name});
            try printExpr(decl.value, writer);
            try writer.writeAll(";\n");
        }

        fn printFnDecl(decl: Ast.FnDeclaration, writer: Writer) Error!void {
            try writer.print("fn {s}(", .{decl.name});

            var first_param = decl.parameters;
            while (first_param) |param| : (first_param = param.next) {
                try writer.print("{s}", .{param.name});
                if (param.next != null) {
                    try writer.writeAll(", ");
                }
            }

            try writer.writeAll(") ");

            try printBlock(decl.body, indent_level, writer);
        }

        fn printBlock(stmt: *Ast.Statement, indent: usize, writer: Writer) Error!void {
            try writer.writeAll("{\n");

            var iter: ?*Ast.Statement = stmt;
            while (iter) |item| : (iter = item.next) {
                try printStatement(item.*, indent + indent_level, writer);
            }

            try printIndent(indent, writer);
            try writer.writeAll("}\n");
        }

        fn printStatement(stmt: Ast.Statement, indent: usize, writer: Writer) Error!void {
            try printIndent(indent, writer);
            switch (stmt.data) {
                .expression => |val| {
                    try printExpr(val.*, writer);
                    try writer.writeAll(";\n");
                },
                .assignment => |val| {
                    try printExpr(val.target.*, writer);
                    try writer.writeAll(" = ");
                    try printExpr(val.value.*, writer);
                    try writer.writeAll(";\n");
                },
                .conditional => |val| {
                    try writer.writeAll("if (");
                    try printExpr(val.condition.*, writer);
                    try writer.writeAll(") ");
                    try printBlock(val.true_branch, indent + indent_level, writer);
                    if (val.false_branch) |false_branch| {
                        try writer.writeAll("else ");
                        try printBlock(false_branch, indent + indent_level, writer);
                    }
                },
                .loop => |val| {
                    try writer.writeAll("while (");
                    try printExpr(val.condition.*, writer);
                    try writer.writeAll(") ");
                    try printBlock(val.body, indent + indent_level, writer);
                },
                .@"break" => try writer.writeAll("break;\n"),
                .@"continue" => try writer.writeAll("continue;\n"),
                .@"return" => try writer.writeAll("return;\n"),
            }
        }

        fn printExpr(expr: Ast.Expression, writer: Writer) Error!void {
            _ = expr;
            try writer.writeAll("<expression>");
        }

        fn printIndent(indent: usize, writer: Writer) Error!void {
            const padding = indent_char.* ** 64;
            var i: usize = 0;
            while (i < indent) {
                const l = std.math.min(padding.len, indent - i);
                try writer.writeAll(padding[0..l]);
                i += l;
            }
        }
    };
}

const TokenType = enum(u8) {
    whitespace,
    comment,

    identifier,
    number,

    @"var",
    @"const",
    @"fn",
    @"and",
    @"or",
    @"continue",
    @"break",
    @"return",
    @"if",
    @"while",

    @",",
    @"=",
    @";",
    @"{",
    @"}",
    @"[",
    @"]",
    @"(",
    @")",
    @"+",
    @"-",
    @"*",
    @"/",
    @"%",
    @"&",
    @"|",
    @"^",
    @">",
    @"<",
    @">=",
    @"<=",
    @"==",
    @">>",
    @">>>",
    @"<<",
    @"!",
    @"~",
};

const Pattern = ptk.Pattern(TokenType);
const Tokenizer = ptk.Tokenizer(TokenType, &.{
    Pattern.create(.whitespace, ptk.matchers.whitespace),
    Pattern.create(.comment, ptk.matchers.sequenceOf(.{ ptk.matchers.literal("#"), ptk.matchers.takeNoneOf("\n"), ptk.matchers.literal("\n") })),
    Pattern.create(.comment, ptk.matchers.sequenceOf(.{ ptk.matchers.literal("#"), ptk.matchers.literal("\n") })),

    Pattern.create(.@"var", ptk.matchers.literal("var")),
    Pattern.create(.@"const", ptk.matchers.literal("const")),
    Pattern.create(.@"fn", ptk.matchers.literal("fn")),
    Pattern.create(.@"and", ptk.matchers.literal("and")),
    Pattern.create(.@"or", ptk.matchers.literal("or")),
    Pattern.create(.@"continue", ptk.matchers.literal("continue")),
    Pattern.create(.@"break", ptk.matchers.literal("break")),
    Pattern.create(.@"return", ptk.matchers.literal("return")),
    Pattern.create(.@"if", ptk.matchers.literal("if")),
    Pattern.create(.@"while", ptk.matchers.literal("while")),

    Pattern.create(.@">=", ptk.matchers.literal(">=")),
    Pattern.create(.@"<=", ptk.matchers.literal("<=")),
    Pattern.create(.@"==", ptk.matchers.literal("==")),
    Pattern.create(.@">>", ptk.matchers.literal(">>")),
    Pattern.create(.@">>>", ptk.matchers.literal(">>>")),
    Pattern.create(.@"<<", ptk.matchers.literal("<<")),

    Pattern.create(.@";", ptk.matchers.literal(";")),
    Pattern.create(.@"{", ptk.matchers.literal("{")),
    Pattern.create(.@"}", ptk.matchers.literal("}")),
    Pattern.create(.@"[", ptk.matchers.literal("[")),
    Pattern.create(.@"]", ptk.matchers.literal("]")),
    Pattern.create(.@"(", ptk.matchers.literal("(")),
    Pattern.create(.@")", ptk.matchers.literal(")")),
    Pattern.create(.@"+", ptk.matchers.literal("+")),
    Pattern.create(.@"-", ptk.matchers.literal("-")),
    Pattern.create(.@"*", ptk.matchers.literal("*")),
    Pattern.create(.@"/", ptk.matchers.literal("/")),
    Pattern.create(.@"%", ptk.matchers.literal("%")),
    Pattern.create(.@"&", ptk.matchers.literal("&")),
    Pattern.create(.@"|", ptk.matchers.literal("|")),
    Pattern.create(.@"^", ptk.matchers.literal("^")),
    Pattern.create(.@">", ptk.matchers.literal(">")),
    Pattern.create(.@"<", ptk.matchers.literal("<")),
    Pattern.create(.@"!", ptk.matchers.literal("!")),
    Pattern.create(.@"~", ptk.matchers.literal("~")),
    Pattern.create(.@"=", ptk.matchers.literal("=")),
    Pattern.create(.@",", ptk.matchers.literal(",")),

    Pattern.create(.identifier, ptk.matchers.identifier),
    Pattern.create(.number, ptk.matchers.decimalNumber),
});

const ParserCore = ptk.ParserCore(Tokenizer, .{ .comment, .whitespace });

const Ast = struct {
    const Self = @This();

    memory: std.heap.ArenaAllocator,
    top_level: ?*TopLevelNode,

    pub fn init(allocator: std.mem.Allocator) Ast {
        return Ast{
            .memory = std.heap.ArenaAllocator.init(allocator),
            .top_level = null,
        };
    }

    pub fn deinit(self: *Self) void {
        self.memory.deinit();
        self.* = undefined;
    }

    fn alloc(self: *Self, comptime T: type) !*T {
        return try self.memory.allocator().create(T);
    }

    pub const TopLevelNode = struct {
        next: ?*@This(),
        data: Data,

        const Data = union(enum) {
            @"var": VarDeclaration,
            @"const": ConstDeclaration,
            @"fn": FnDeclaration,
        };
    };

    pub const VarDeclaration = struct {
        name: []const u8,
        value: ?Expression,
    };

    pub const ConstDeclaration = struct {
        name: []const u8,
        value: Expression,
    };

    pub const FnDeclaration = struct {
        name: []const u8,
        parameters: ?*Parameter,
        body: *Statement,
    };

    pub const Parameter = struct {
        name: []const u8,
        next: ?*Parameter,
    };

    pub const Statement = struct {
        data: Data,
        next: ?*Statement,

        const Data = union(enum) {
            expression: *Expression,
            assignment: Assignment,
            conditional: Conditional,
            loop: Loop,
            @"break",
            @"continue",
            @"return": ?*Expression,
        };
    };

    pub const Assignment = struct {
        target: *Expression,
        value: *Expression,
    };

    pub const Conditional = struct {
        condition: *Expression,
        true_branch: *Statement,
        false_branch: ?*Statement,
    };

    pub const Loop = struct {
        condition: *Expression,
        body: *Statement,
    };

    pub const Expression = union(enum) {
        number: []const u8,
        identifier: []const u8,
    };
};
