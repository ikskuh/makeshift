# Linkers and Loaders

## Requirements

Module format should store the already generated machine code by the compiler exept for the final addresses. It should also store information about the symbol names of the final adresses as well as a list of exported symbols with start address and symbol size.

## File Format

All integers are in little endian, there is no automatic padding between fields.

### Header

```zig
magic: [4]u8 = .{ 0xFB, 0xAD, 0xB6, 0x02 },
export_table:  u32,
import_table:  u32,
string_table:  u32,
section_start: u32,
section_size:  u32,
symbol_size:   u8,  // is one of { 1, 2, 4, 8 }
```

- `export_table`: The byte offset into the file where the exports are listed. If `0`, no export table exists.
- `import_table`: The byte offset into the file where the imports are listed. If `0`, no import table exists.
- `string_table`: The byte offset into the file where the string table is located. If `0`, no string table exists. Must be present, if either `export_table` or `import_table` exists.
- `section_start`: The byte offset into the file where the memory section starts. All section offsets are relative to this offset.
- `section_size`: The number of bytes in the memory section.
- `symbol_size` specifies the pointer size of this module. This means that each symbol has pointer size of `symbol_size` bytes.

### Export/Import Table

The import table specifies which external symbols are referenced at which place. For this, each entry in the table will patch the external symbol `symbol_name` into the finaly binary at module offset `offset`.

The export table specifies which symbols this module provides at which offsets. For each entry, a symbol named `symbol_name` is located at module start + `offset`.

Both tables use the same data structure:

```zig
count: u32,
entries: [count]struct {
  symbol_name: u32, // offset into the string table
  offset:      u32, // offset into the section where the symbol is located/referenced
},
```

### String Table

The string table is composed of several byte strings. The string table starts with a declaration about how large the table in bytes is.

```zig
size: u32,
data: ???, // the data is variably sized, see below
```

Strings are referenced by the offset into the symbol table, so the first string is located at `0x04`. Each string looks like this:

```zig
length: u32,
bytes:  [length]u8,
zero:   u8 = 0,
```
