import os
import csv


def path_test(path: str, extensions: list) -> str:
    """Validate that a given path points to a readable file with extension.

    Resolves the provided path to an absolute path and runs a series of
    checks to ensure the file exists, is a regular file (not a directory),
    is readable by the current user, and has a .jpg or .jpeg extension.

    Args:
        path: A relative or absolute filesystem path to validate.

    Returns:
        The resolved absolute path to the validated file.

    Raises:
        AssertionError: If the path does not exist, is not a regular file,
            is not readable, or does not have a correct extension.
    """
    abspath = os.path.abspath(path)
    assert os.path.exists(abspath), f"Wrong Path {path}"
    assert os.path.isfile(abspath), f"{path} is not a file"
    assert os.access(abspath, os.R_OK), f"User can not read permit on {path}"
    _, ext = os.path.splitext(abspath)
    assert (
        ext.lower() in extensions
    ), f"Expected a file with one extension like {extensions}, got '{ext[1:]}'"
    return abspath


def ft_load_csv(path: str) -> list[list[float]]:
    """Load a CSV file from disk and return it as list of list.

    Validates the given path using path_test, then opens the image with
    Pillow and converts it to a NumPy array. Prints the array's shape
    (height, width, channels) to stdout before returning.

    Args:
        path: A relative or absolute filesystem path to a JPEG image.

    Returns:
        A NumPy array of shape (height, width, channels) representing
        the pixel data of the image.

    Raises:
        AssertionError: If path validation fails (see path_test).
    """
    abspath = path_test(path, [".csv", ".txt"])
    with open(abspath, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        data = [row for row in reader]

    num_rows = len(data)
    num_cols = len(data[0]) if data else 0

    print(f"CSV file has {num_rows} records with {num_cols} features")
    return data
