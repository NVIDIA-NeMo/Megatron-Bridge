from typing import List, Optional, Tuple

from megatron.hub.data.loaders import get_blend_and_blend_per_split


_BLEND_TYPE = Optional[Tuple[List[str], Optional[List[float]]]]
_BLEND_PER_SPLIT_TYPE = Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
_SPLIT_TYPE = Optional[str]


def get_blend_fields_from_data_paths(
    data_paths: Optional[List[str]] = None,
    data_args_path: Optional[str] = None,
    train_data_path: Optional[List[str]] = None,
    valid_data_path: Optional[List[str]] = None,
    test_data_path: Optional[List[str]] = None,
    per_split_data_args_path: Optional[str] = None,
    mock: bool = False,
) -> Tuple[_BLEND_TYPE, _BLEND_PER_SPLIT_TYPE, _SPLIT_TYPE]:
    """
    Common configuration logic for blend, blend_per_split, split dataset config fields.

    Handles mock and real data.

    Args:
        data_paths (Optional[List[str]]): List of paths to dataset files. If None, mock data will be used.
        data_args_path (Optional[str]): Path to file containing data arguments.
        train_data_path (Optional[List[str]]): List of training data paths.
        valid_data_path (Optional[List[str]]): List of validation data paths.
        test_data_path (Optional[List[str]]): List of test data paths.
        per_split_data_args_path (Optional[str]): Path to JSON file with per-split data configuration.
        mock (bool): Whether to use mock data. If True, ignores data_paths.

    Returns:
        A tuple (blend, blend_per_split, split), the corresponding fields to be passed to GPTDatasetConfig.
    """
    has_any_data_config = any(
        [data_paths, data_args_path, train_data_path, valid_data_path, test_data_path, per_split_data_args_path]
    )

    if mock or not has_any_data_config:
        # Mock data configuration
        blend = None  # Will trigger mock mode automatically
        blend_per_split = None  # Will trigger mock mode automatically
        split = "1,1,1"  # Equal splits for testing
    else:
        # Real data configuration
        blend_weights, blend_per_split_weights = get_blend_and_blend_per_split(
            data_paths=data_paths,
            data_args_path=data_args_path,
            train_data_paths=train_data_path,
            valid_data_paths=valid_data_path,
            test_data_paths=test_data_path,
            per_split_data_args_path=per_split_data_args_path,
        )

        if blend_weights is None and blend_per_split_weights is None:
            # No data provided, fall back to mock mode
            blend = None
            blend_per_split = None
            split = "1,1,1"
        else:
            # Construct data_path from the inputs
            if data_paths is not None:
                data_path = data_paths
            elif data_args_path is not None:
                data_path = data_args_path
            else:
                data_path = []
                if train_data_path:
                    data_path.extend(train_data_path)
                if valid_data_path:
                    data_path.extend(valid_data_path)
                if test_data_path:
                    data_path.extend(test_data_path)
                if per_split_data_args_path:
                    data_path = per_split_data_args_path

            # Create the tuples expected by BlendedMegatronDatasetConfig
            # Prioritize blend_per_split_weights over blend_weights if both are provided
            if blend_per_split_weights is not None:
                # For per-split, we need to construct the paths for each split
                train_paths = train_data_path or []
                valid_paths = valid_data_path or []
                test_paths = test_data_path or []

                blend_per_split = [
                    (train_paths, blend_per_split_weights[0]) if train_paths else None,
                    (valid_paths, blend_per_split_weights[1]) if valid_paths else None,
                    (test_paths, blend_per_split_weights[2]) if test_paths else None,
                ]
                # When using blend_per_split, split should be None and blend should be None
                split = None
                blend = None
            elif blend_weights is not None:
                blend = (data_path if isinstance(data_path, list) else [data_path], blend_weights)
                blend_per_split = None
                # When using regular blend, we can use split
                split = "9999,8,2"
            else:
                blend = None
                blend_per_split = None
                split = "9999,8,2"

    return blend, blend_per_split, split
