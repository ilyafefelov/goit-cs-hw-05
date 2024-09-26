import argparse
import asyncio
import logging
import os
from pathlib import Path

import aiofiles
import aiofiles.os
import aiofiles.ospath

# Configure logging
logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s %(levelname)s:%(message)s",
    filename="file_sorter.log",
    filemode="w",
)


async def copy_file(src_file: Path, dest_dir: Path):
    """
    Asynchronously copies a file to the destination directory, placing it in a subdirectory
    named after the file's extension.
    """
    try:
        # Get the file extension and create the target directory
        ext = src_file.suffix[1:] if src_file.suffix else "no_extension"
        target_dir = dest_dir / ext
        await aiofiles.os.makedirs(target_dir, exist_ok=True)

        # Define the destination file path
        dest_file = target_dir / src_file.name

        # Copy the file asynchronously
        async with aiofiles.open(src_file, "rb") as fsrc:
            async with aiofiles.open(dest_file, "wb") as fdst:
                while True:
                    chunk = await fsrc.read(1024 * 1024)  # Read in 1MB chunks
                    if not chunk:
                        break
                    await fdst.write(chunk)
    except Exception as e:
        logging.error(f"Error copying file {src_file} to {dest_file}: {e}")


async def read_folder(src_dir: Path, dest_dir: Path):
    """
    Asynchronously reads all files in the source directory and its subdirectories,
    and schedules them for copying.
    """
    tasks = []
    try:
        for root, _, files in os.walk(src_dir):
            for file in files:
                src_file = Path(root) / file
                task = asyncio.create_task(copy_file(src_file, dest_dir))
                tasks.append(task)
    except Exception as e:
        logging.error(f"Error reading folder {src_dir}: {e}")

    await asyncio.gather(*tasks)


def main():
    # Argument parser for command-line arguments
    parser = argparse.ArgumentParser(
        description="Asynchronously sort files based on extension."
    )
    parser.add_argument("source", type=str, help="Path to the source folder.")
    parser.add_argument("destination", type=str, help="Path to the destination folder.")

    args = parser.parse_args()

    # Initialize asynchronous paths
    src_dir = Path(args.source).resolve()
    dest_dir = Path(args.destination).resolve()

    # Check if source directory exists
    if not src_dir.is_dir():
        print(f"The source directory '{src_dir}' does not exist.")
        return

    # Run the asynchronous read_folder function
    asyncio.run(read_folder(src_dir, dest_dir))


if __name__ == "__main__":
    main()
