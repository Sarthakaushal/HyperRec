def print_banner(text):
    print("\n" + "=" * len(text))
    print(text)
    print("=" * len(text) + "\n")
    

class UnsupportedFileFormatError(Exception):
    """Raised when the input file format is not supported."""
    pass