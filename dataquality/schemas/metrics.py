from typing import Dict, List, Optional

from pydantic import BaseModel, Field, StrictStr, root_validator


class HashableBaseModel(BaseModel):
    """Hashable BaseModel https://github.com/pydantic/pydantic/issues/1303"""

    def __hash__(self) -> int:
        hashable_vals = [i for i in self.__dict__.values() if not isinstance(i, list)]
        unhashables = [tuple(i) for i in self.__dict__.values() if isinstance(i, list)]
        return hash((type(self),) + tuple(hashable_vals + unhashables))


class MetaFilter(HashableBaseModel):
    """A class for filtering arbitrary metadata dataframe columns

    For example, to filter on a logged metadata column, "is_happy" for values [True],
    you can create a MetaFilter(name="is_happy", isin=[True])

    You can use this filter for any columns, not just metadata columns.
    For example, you can use this to filter for DEP scores above 0.5:
    MetaFilter(name="data_error_potential", greater_than=0.5)
    """

    name: StrictStr
    greater_than: Optional[float] = None
    less_than: Optional[float] = None
    isin: Optional[List[str]] = None


class InferenceFilter(HashableBaseModel):
    """A class for filtering an inference split

    - `is_otb`: Filters samples that are / are not On-The-Boundary
    - `is_drifted`: Filters samples that are / are not Drifted
    """

    is_otb: Optional[bool] = None
    is_drifted: Optional[bool] = None


class LassoSelection(HashableBaseModel):
    """Representation of a lasso selection (used during an embeddings selection)

    x and y correspond to the cursor movement while tracing the lasso. This is natively
    provided by plotly when creating a lasso selection
    """

    x: List[float]
    y: List[float]

    @root_validator()
    def validate_xy(cls: BaseModel, values: Dict[str, List]) -> Dict[str, List]:
        if len(values.get("x", [])) != len(values.get("y", [])):
            raise ValueError("x and y must have the same number of points")
        if len(values.get("x", [])) < 1:
            raise ValueError("x and y must have at least 1 value")
        return values


class FilterParams(HashableBaseModel):
    """A class for sending filters to the API alongside most any request.

    Each field represents things you can filter the dataframe on.

    Args:
        ids: List[int] = []  filter for specific IDs in the dataframe (span IDs for NER)
        similar_to: Optional[int] = None  provide an ID to run similarity search on
        num_similar_to: Optional[int] = None  if running similarity search, how many
        text_pat: Optional[StrictStr] = None  filter text samples by some text pattern
        regex: Optional[bool] = None  if searching with text, whether to use regex
        data_error_potential_high: Optional[float] = None  only samples with DEP <= this
        data_error_potential_low: Optional[float] = None  only samples with DEP >= this
        misclassified_only: Optional[bool] = None  Only look at missed samples
        gold_filter: Optional[List[StrictStr]] = None  filter GT classes
        pred_filter: Optional[List[StrictStr]] = None  filter prediction classes
        meta_filter: Optional[List[MetaFilter]] = None  see MetaFilter class
        inference_filter: Optional[InferenceFilter] = None  see InferenceFilter class
        span_sample_ids: Optional[List[int]] = None  (NER only) filter for full samples
        span_text: Optional[str] = None  (NER only) filter only on span text
        exclude_ids: List[int] = []  opposite of `ids`
        lasso: Optional[LassoSelection] = None  see LassoSelection class
        class_filter: Optional[List[StrictStr]] = None  filter GT OR prediction
        likely_mislabeled: Optional[List[StrictStr]] = None  Filter for only
            likely_mislabeled samples. False/None will return all samples
        likely_mislabeled_dep_percentile: Optional[int] A percentile threshold for l
            ikely mislabeled. This field (ranged 0-100) determines the precision of the
            likely_mislabeled filter. The threshold is applied against the DEP
            distribution of the likely_mislabeled samples. A threshold of 0 returns all,
            100 returns 1 sample, and 50 will return the top 50% DEP samples that are
            likely_mislabeled. Higher = more precision, lower = more recall. Default 50.
    """

    ids: List[int] = []
    similar_to: Optional[List[int]] = None
    num_similar_to: Optional[int] = None
    text_pat: Optional[StrictStr] = None
    regex: Optional[bool] = None
    data_error_potential_high: Optional[float] = None
    data_error_potential_low: Optional[float] = None
    misclassified_only: Optional[bool] = None
    gold_filter: Optional[List[StrictStr]] = None
    pred_filter: Optional[List[StrictStr]] = None
    meta_filter: Optional[List[MetaFilter]] = None
    inference_filter: Optional[InferenceFilter] = None
    span_sample_ids: Optional[List[int]] = None
    span_text: Optional[str] = None
    span_regex: Optional[bool] = None
    exclude_ids: List[int] = []
    lasso: Optional[LassoSelection] = None
    class_filter: Optional[List[StrictStr]] = None
    likely_mislabeled: Optional[bool] = None
    likely_mislabeled_dep_percentile: Optional[int] = Field(50, ge=0, le=100)
