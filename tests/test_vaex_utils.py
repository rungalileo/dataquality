import vaex

from dataquality.utils.vaex import drop_empty_columns


def test_drop_empty_cols() -> None:
    df1 = vaex.example()[["x", "y", "z"]]
    df1["split"] = vaex.vconstant("training", length=len(df1))
    df2 = vaex.example()[["vx", "vy", "x", "y"]]
    df2["split"] = vaex.vconstant("test", length=len(df2))

    in_frame = vaex.concat([df2, df1])
    split_df = in_frame[in_frame["split"] == "test"]
    assert split_df["z"].ismissing().to_numpy().all()
    split_df = drop_empty_columns(split_df)
    assert "z" not in split_df.get_column_names()
