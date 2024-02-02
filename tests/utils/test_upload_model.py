import os
import tarfile
import tempfile

from dataquality.utils.upload_model import create_tar_archive


def test_create_tar_archive() -> None:
    # Create a temporary directory as the source folder
    with tempfile.TemporaryDirectory() as source_folder:
        # Populate the source folder with mock files
        with open(os.path.join(source_folder, "file1.txt"), "w") as f:
            f.write("This is a mock file.")
        with open(os.path.join(source_folder, "file2.txt"), "w") as f:
            f.write("This is another mock file.")

        # Create a subfolder with another mock file
        subfolder_path = os.path.join(source_folder, "subfolder")
        os.mkdir(subfolder_path)
        with open(os.path.join(subfolder_path, "file3.txt"), "w") as f:
            f.write("This is a mock file inside a subfolder.")
        tar_filename = ""
        # Create a temporary file to hold the tar archive
        with tempfile.NamedTemporaryFile() as tar_file:
            tar_filename = tar_file.name

            # Call the function to create a tar archive
            create_tar_archive(source_folder, tar_filename)

            # Untar the created archive to a new temporary directory
            with tempfile.TemporaryDirectory() as untar_folder:
                with tarfile.open(tar_filename, "r") as archive:
                    archive.extractall(path=untar_folder)

                # Verify the contents of the untarred directory match the original
                assert os.path.exists(os.path.join(untar_folder, "file1.txt"))
                assert os.path.exists(os.path.join(untar_folder, "file2.txt"))
                assert os.path.exists(
                    os.path.join(untar_folder, "subfolder", "file3.txt")
                )

        assert not os.path.exists(tar_filename)
