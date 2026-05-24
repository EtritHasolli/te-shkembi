import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


RANDOM_STATE_INT = 42
TRAIN_SIZE_FLOAT = 0.70
VALIDATION_SIZE_FLOAT = 0.15
TEST_SIZE_FLOAT = 0.15

IMAGE_EXTENSIONS_SET = {".jpg", ".jpeg", ".png"}

GENDER_LABELS_DICT = {
    0: "male",
    1: "female",
}

RACE_LABELS_DICT = {
    0: "white",
    1: "black",
    2: "asian",
    3: "indian",
    4: "others",
}

CATEGORIZE_TYPE_NAME_LIST = [ "teen", "adult", "senior" ]


@dataclass
class DatasetRecord:
    image_path_str: str
    filename_str: str
    age_int: int
    gender_int: int
    gender_label_str: str
    race_int: int
    race_label_str: str
    age_group_int: int
    age_group_label_str: str


class UTKFaceDataset(Dataset):
    def __init__(self, metadata_path_str: str, transform_obj=None):
        self.metadata_df = pd.read_csv(metadata_path_str)
        self.transform_obj = transform_obj

    def __len__(self) -> int:
        return len(self.metadata_df)

    def __getitem__(self, index_int: int):
        row_obj = self.metadata_df.iloc[index_int]
        image_obj = Image.open(row_obj["image_path"]).convert("RGB")

        if self.transform_obj:
            image_obj = self.transform_obj(image_obj)

        target_dict = {
            "age": int(row_obj["age"]),
            "gender": int(row_obj["gender"]),
            "race": int(row_obj["race"]),
            "age_group": int(row_obj["age_group"]),
        }

        return image_obj, target_dict


class UTKFacePreProcessor:
    def __init__(
        self,
        input_dir_str: str,
        output_dir_str: str = "pre-processed2",
        image_size_int: Optional[int] = None,
    ):
        self.input_path_obj = Path(input_dir_str)
        self.output_path_obj = Path(output_dir_str)
        self.image_size_int = image_size_int

    def _log_step(self, step_num_int: int, total_steps_int: int, message_str: str) -> None:
        print(f"[{step_num_int}/{total_steps_int}] {message_str}")

    def _log_info(self, message_str: str) -> None:
        print(f"[INFO] {message_str}")

    def _log_ok(self, message_str: str) -> None:
        print(f"[OK] {message_str}")

    def _log_error(self, message_str: str) -> None:
        print(f"[ERROR] {message_str}", file=sys.stderr)

    def run(self) -> None:
        total_steps_int = 5

        try:
            self._log_step(1, total_steps_int, "Validating input directory")
            self._validate_input_dir()
            self._log_ok("Input directory is valid.")

            self._log_step(2, total_steps_int, "Preparing output directory")
            self._reset_output_dir()
            self._log_ok(f"Output directory ready: {self.output_path_obj}")

            self._log_step(3, total_steps_int, "Scanning and validating images")
            records_list, skipped_dict = self._collect_valid_records()
            dataset_df = pd.DataFrame([record_obj.__dict__ for record_obj in records_list])

            dataset_df = dataset_df.rename(
                columns={
                    "image_path_str": "image_path",
                    "filename_str": "filename",
                    "age_int": "age",
                    "gender_int": "gender",
                    "gender_label_str": "gender_label",
                    "race_int": "race",
                    "race_label_str": "race_label",
                    "age_group_int": "age_group",
                    "age_group_label_str": "age_group_label",
                }
            )
            self._log_ok(f"Valid images collected: {len(dataset_df)}")

            self._log_step(4, total_steps_int, "Splitting dataset and writing files")
            train_df, validation_df, test_df = self._split_dataset(dataset_df)

            self._save_split("train", train_df)
            self._save_split("valid", validation_df)
            self._save_split("test", test_df)
            self._log_ok("Dataset splits saved.")

            self._log_step(5, total_steps_int, "Writing summary")
            summary_dict = self._build_summary(dataset_df, train_df, validation_df, test_df, skipped_dict)
            self._save_summary(summary_dict)
            self._log_ok("Summary saved.")

            self._log_ok("Pre-processing completed successfully.")
            print("Summary")
            print(f"  Input folder: {self.input_path_obj}")
            print(f"  Output folder: {self.output_path_obj}")
            print(f"  Total valid images: {len(dataset_df)}")
            print(f"  Train images: {len(train_df)}")
            print(f"  Valid images: {len(validation_df)}")
            print(f"  Test images: {len(test_df)}")
            print(f"  Invalid filenames skipped: {skipped_dict['invalid_filename_count']}")
            print(f"  Unreadable images skipped: {skipped_dict['unreadable_image_count']}")
        except Exception as exc:
            self._log_error(f"{type(exc).__name__}: {exc}")
            raise

    def _validate_input_dir(self) -> None:
        if not self.input_path_obj.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_path_obj}")

        if not self.input_path_obj.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {self.input_path_obj}")

    def _reset_output_dir(self) -> None:
        if self.output_path_obj.exists():
            shutil.rmtree(self.output_path_obj)

        self.output_path_obj.mkdir(parents=True, exist_ok=True)

    def _collect_valid_records(self) -> Tuple[List[DatasetRecord], Dict[str, int]]:
        records_list: List[DatasetRecord] = []
        invalid_filename_count_int = 0
        unreadable_image_count_int = 0

        image_paths_list = sorted(
            [
                path_obj
                for path_obj in self.input_path_obj.rglob("*")
                if path_obj.suffix.lower() in IMAGE_EXTENSIONS_SET
            ]
        )

        total_images_int = len(image_paths_list)
        progress_every_int = 1000

        self._log_info(f"Found {total_images_int} image files.")

        for index_int, image_path_obj in enumerate(image_paths_list, start=1):
            parsed_label_tuple = self._parse_filename(image_path_obj.name)

            if parsed_label_tuple is None:
                invalid_filename_count_int += 1
            elif not self._is_readable_image(image_path_obj):
                unreadable_image_count_int += 1
            else:
                age_int, gender_int, race_int = parsed_label_tuple
                age_group_int, age_group_label_str = self._get_age_group(age_int)

                records_list.append(
                    DatasetRecord(
                        image_path_str=str(image_path_obj),
                        filename_str=image_path_obj.name,
                        age_int=age_int,
                        gender_int=gender_int,
                        gender_label_str=GENDER_LABELS_DICT[gender_int],
                        race_int=race_int,
                        race_label_str=RACE_LABELS_DICT[race_int],
                        age_group_int=age_group_int,
                        age_group_label_str=age_group_label_str,
                    )
                )

            if index_int % progress_every_int == 0 or index_int == total_images_int:
                self._log_info(
                    "Progress: "
                    f"{index_int}/{total_images_int} files scanned | "
                    f"valid: {len(records_list)} | "
                    f"invalid: {invalid_filename_count_int} | "
                    f"unreadable: {unreadable_image_count_int}"
                )

        skipped_dict = {
            "invalid_filename_count": invalid_filename_count_int,
            "unreadable_image_count": unreadable_image_count_int,
        }

        if len(records_list) == 0:
            raise ValueError("No valid UTKFace images were found.")

        return records_list, skipped_dict

    def _parse_filename(self, filename_str: str) -> Optional[Tuple[int, int, int]]:
        filename_parts_list = filename_str.split("_")

        if len(filename_parts_list) < 4:
            return None

        try:
            age_int = int(filename_parts_list[0])
            gender_int = int(filename_parts_list[1])
            race_int = int(filename_parts_list[2])
        except ValueError:
            return None

        if age_int < 0 or age_int > 116:
            return None

        if gender_int not in GENDER_LABELS_DICT:
            return None

        if race_int not in RACE_LABELS_DICT:
            return None

        return age_int, gender_int, race_int

    def _is_readable_image(self, image_path_obj: Path) -> bool:
        try:
            with Image.open(image_path_obj) as image_obj:
                image_obj.verify()
            return True
        except (UnidentifiedImageError, OSError):
            return False

    def _get_age_group(self, age_int: int) -> Tuple[int, str]:
        # Map to 3 class folders: young (0-19), adult (20-59), old (60+).
        if age_int <= 19:
            return 0, CATEGORIZE_TYPE_NAME_LIST[0]

        if age_int <= 59:
            return 1, CATEGORIZE_TYPE_NAME_LIST[1]

        return 2, CATEGORIZE_TYPE_NAME_LIST[2]

    def _split_dataset(self, dataset_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        stratify_series = dataset_df["gender"].astype(str) + "_" + dataset_df["race"].astype(str)

        train_df, temp_df = train_test_split(
            dataset_df,
            train_size=TRAIN_SIZE_FLOAT,
            random_state=RANDOM_STATE_INT,
            stratify=stratify_series,
        )

        temp_stratify_series = temp_df["gender"].astype(str) + "_" + temp_df["race"].astype(str)
        validation_ratio_float = VALIDATION_SIZE_FLOAT / (VALIDATION_SIZE_FLOAT + TEST_SIZE_FLOAT)

        validation_df, test_df = train_test_split(
            temp_df,
            train_size=validation_ratio_float,
            random_state=RANDOM_STATE_INT,
            stratify=temp_stratify_series,
        )

        return train_df.reset_index(drop=True), validation_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def _save_split(self, split_name_str: str, split_df: pd.DataFrame) -> None:
        split_path_obj = self.output_path_obj / split_name_str
        split_path_obj.mkdir(parents=True, exist_ok=True)

        output_rows_list = []

        for index_int, row_obj in split_df.iterrows():

            # Use gender labels instead of age groups
            gender_label_str = row_obj["gender_label"]

            # Create:
            # train/male
            # train/female
            class_path_obj = split_path_obj / gender_label_str
            class_path_obj.mkdir(parents=True, exist_ok=True)

            source_image_path_obj = Path(row_obj["image_path"])

            output_filename_str = (
                f"{index_int:06d}_{source_image_path_obj.name}"
            )

            output_image_path_obj = (
                class_path_obj / output_filename_str
            )

            self._save_image(
                source_image_path_obj,
                output_image_path_obj
            )

            row_dict = row_obj.to_dict()

            row_dict["original_image_path"] = row_dict["image_path"]

            # Update image path to new processed image
            row_dict["image_path"] = str(output_image_path_obj)

            row_dict["split"] = split_name_str

            output_rows_list.append(row_dict)

        output_df = pd.DataFrame(output_rows_list)

        output_df.to_csv(
            split_path_obj / "metadata.csv",
            index=False
    )

    def _save_image(self, source_image_path_obj: Path, output_image_path_obj: Path) -> None:
        with Image.open(source_image_path_obj) as image_obj:
            image_obj = image_obj.convert("RGB")

            if self.image_size_int is not None:
                image_obj = image_obj.resize((self.image_size_int, self.image_size_int))

            image_obj.save(output_image_path_obj, quality=95)

    def _build_summary(
        self,
        dataset_df: pd.DataFrame,
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        test_df: pd.DataFrame,
        skipped_dict: Dict[str, int],
    ) -> Dict:
        return {
            "input_dir": str(self.input_path_obj),
            "output_dir": str(self.output_path_obj),
            "image_size": self.image_size_int if self.image_size_int is not None else "original",
            "total_valid_images": int(len(dataset_df)),
            "train_images": int(len(train_df)),
            "valid_images": int(len(validation_df)),
            "test_images": int(len(test_df)),
            "train_ratio": TRAIN_SIZE_FLOAT,
            "valid_ratio": VALIDATION_SIZE_FLOAT,
            "test_ratio": TEST_SIZE_FLOAT,
            "gender_distribution": dataset_df["gender_label"].value_counts().to_dict(),
            "race_distribution": dataset_df["race_label"].value_counts().to_dict(),
            "age_group_distribution": dataset_df["age_group_label"].value_counts().to_dict(),
            "skipped": skipped_dict,
        }

    def _save_summary(self, summary_dict: Dict) -> None:
        summary_path_obj = self.output_path_obj / "dataset_summary.json"

        with open(summary_path_obj, "w", encoding="utf-8") as file_obj:
            json.dump(summary_dict, file_obj, indent=4)


def build_transforms(image_size_int: Optional[int] = None):
    train_transforms_list = []

    if image_size_int is not None:
        train_transforms_list.append(transforms.Resize((image_size_int, image_size_int)))

    train_transforms_list.extend(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    evaluation_transforms_list = []

    if image_size_int is not None:
        evaluation_transforms_list.append(transforms.Resize((image_size_int, image_size_int)))

    evaluation_transforms_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    return transforms.Compose(train_transforms_list), transforms.Compose(evaluation_transforms_list)


def prepare_data_loaders(
    processed_dir_str: str = "pre-processed2",
    batch_size_int: int = 32,
    image_size_int: Optional[int] = None,
    num_workers_int: int = 2,
) -> Dict[str, DataLoader]:
    processed_path_obj = Path(processed_dir_str)

    train_transform_obj, evaluation_transform_obj = build_transforms(image_size_int=image_size_int)

    train_dataset_obj = UTKFaceDataset(
        metadata_path_str=str(processed_path_obj / "train" / "metadata.csv"),
        transform_obj=train_transform_obj,
    )
    validation_dataset_obj = UTKFaceDataset(
        metadata_path_str=str(processed_path_obj / "valid" / "metadata.csv"),
        transform_obj=evaluation_transform_obj,
    )
    test_dataset_obj = UTKFaceDataset(
        metadata_path_str=str(processed_path_obj / "test" / "metadata.csv"),
        transform_obj=evaluation_transform_obj,
    )

    return {
        "train": DataLoader(
            train_dataset_obj,
            batch_size=batch_size_int,
            shuffle=True,
            num_workers=num_workers_int,
        ),
        "validation": DataLoader(
            validation_dataset_obj,
            batch_size=batch_size_int,
            shuffle=False,
            num_workers=num_workers_int,
        ),
        "test": DataLoader(
            test_dataset_obj,
            batch_size=batch_size_int,
            shuffle=False,
            num_workers=num_workers_int,
        ),
    }


def parse_arguments():
    parser_obj = argparse.ArgumentParser(
        description="Pre-process UTKFace dataset and create train/valid/test splits."
    )

    parser_obj.add_argument(
        "input_dir",
        type=str,
        help="Path to curated dataset folder. Example: curated-dataset",
    )

    parser_obj.add_argument(
        "image_size",
        type=int,
        nargs="?",
        default=None,
        help="Optional resize size. Example: 224 creates 224x224 images. If omitted, original image size is kept.",
    )

    parser_obj.add_argument(
        "--output-dir",
        type=str,
        default="pre-processed2",
        help="Output folder name. Default: pre-processed2",
    )

    return parser_obj.parse_args()


def main() -> None:
    args_obj = parse_arguments()

    processor_obj = UTKFacePreProcessor(
        input_dir_str=args_obj.input_dir,
        output_dir_str=args_obj.output_dir,
        image_size_int=args_obj.image_size,
    )

    processor_obj.run()


if __name__ == "__main__":
    main()
