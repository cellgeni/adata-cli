"""Text from a store or the command line, put into Rich markup.

Everything the CLI prints goes through Rich, which reads `[word]` as a style
tag. Member names, labels, paths and attribute values are data, and scanpy's
own QC column is `pct_counts[mt]`: printed as it is, it shows as
`pct_counts`, and a value holding `[/x]` stops the command. Every such value
goes into a message, status line, progress description or tree label through
`literal`.
"""

from __future__ import annotations

from rich.markup import escape


def literal(value: object) -> str:
    """`value` as markup that prints it as it is, whatever follows it.

    `rich.markup.escape` doubles a trailing backslash, so that a tag after it
    is still read as a tag; where text follows instead, the doubled one is
    printed. Here trailing backslashes are doubled and followed by an empty
    span, which reads the same before a tag and before text.
    """
    text = str(value)
    body = text.rstrip("\\")
    trailing = len(text) - len(body)
    if not trailing:
        return escape(body)
    return escape(body) + "\\" * (2 * trailing) + "[i][/i]"
