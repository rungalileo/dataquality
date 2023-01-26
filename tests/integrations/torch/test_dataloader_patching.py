import torch
from torch.utils.data import DataLoader

from dataquality.schemas.torch import HelperData
from dataquality.utils.torch import patch_dataloaders, unpatch


def test_mp():
    a = torch.arange(0, 10)
    store = {HelperData.dl_next_idx_ids: []}
    mp_dataloader = DataLoader(
        a,
        batch_size=3,
        num_workers=2,
        persistent_workers=True,
        shuffle=True,
    )
    patch_dataloaders(store)
    batches = []
    for batch in mp_dataloader:
        batches.append(batch.long())
    assert len(batches) == len(
        store[HelperData.dl_next_idx_ids]
    ), "number of batches is not the same (part 1)"
    # assert all indices are the same

    for batch, indices in zip(batches, store[HelperData.dl_next_idx_ids]):
        assert torch.LongTensor(batch).equal(
            torch.LongTensor(indices)
        ), "indices are not the same"

    unpatch(store[HelperData.patches])
    store = {HelperData.dl_next_idx_ids: []}
    sp_dataloader = DataLoader(
        a,
        batch_size=3,
        num_workers=0,
        shuffle=True,
    )

    batches = []
    for batch in sp_dataloader:
        batches.append(batch.long())

    assert len(batches) != len(
        store[HelperData.dl_next_idx_ids]
    ), "number of batches is not the same (part 2)"
    # assert all indices are the same
    for batch, indices in zip(batches, store[HelperData.dl_next_idx_ids]):
        print(batch, indices)
        assert not torch.LongTensor(batch).equal(
            torch.LongTensor(indices)
        ), "indices are the same"

    patch_dataloaders(store)
    batches = []
    for batch in sp_dataloader:
        batches.append(batch.long())
    assert len(batches) == len(
        store[HelperData.dl_next_idx_ids]
    ), "number of batches is not the same (part 3)"
    # assert all indices are the same
    for batch, indices in zip(batches, store[HelperData.dl_next_idx_ids]):
        assert torch.LongTensor(batch).equal(
            torch.LongTensor(indices)
        ), "indices are not the same"

    mp_dataloader = DataLoader(
        a,
        batch_size=3,
        num_workers=2,
        persistent_workers=True,
        shuffle=True,
    )

    patch_dataloaders(store)
    store[HelperData.dl_next_idx_ids] = []
    batches = []
    for batch in mp_dataloader:
        batches.append(batch.long())
    assert len(batches) == len(
        store[HelperData.dl_next_idx_ids]
    ), "number of batches is not the same  (part 4)"
    # assert all indices are the same
    for batch, indices in zip(batches, store[HelperData.dl_next_idx_ids]):
        assert torch.LongTensor(batch).equal(
            torch.LongTensor(indices)
        ), "indices are not the same"
    unpatch(store[HelperData.patches])


def test_sp():
    a = torch.arange(0, 10)
    store = {HelperData.dl_next_idx_ids: []}
    mp_dataloader = DataLoader(
        a,
        batch_size=3,
        num_workers=1,
        persistent_workers=True,
        shuffle=True,
    )

    patch_dataloaders(store)
    batches = []
    for batch in mp_dataloader:
        batches.append(batch.long())

    assert len(batches) == len(
        store[HelperData.dl_next_idx_ids]
    ), "number of batches is not the same"
    # assert all indices are the same
    for batch, indices in zip(batches, store[HelperData.dl_next_idx_ids]):
        assert torch.LongTensor(batch).equal(
            torch.LongTensor(indices)
        ), "indices are not the same"

    unpatch(store[HelperData.patches])
    store = {HelperData.dl_next_idx_ids: []}
    sp_dataloader = DataLoader(
        a,
        batch_size=3,
        num_workers=0,
        shuffle=True,
    )

    batches = []
    for batch in sp_dataloader:
        batches.append(batch.long())

    assert len(batches) != len(
        store[HelperData.dl_next_idx_ids]
    ), "number of batches is not the same"
    # assert all indices are the same
    for batch, indices in zip(batches, store[HelperData.dl_next_idx_ids]):
        assert not torch.LongTensor(batch).equal(
            torch.LongTensor(indices)
        ), "indices are the same"

    patch_dataloaders(store)
    batches = []
    for batch in sp_dataloader:
        batches.append(batch.long())
    assert len(batches) == len(
        store[HelperData.dl_next_idx_ids]
    ), "number of batches is not the same"
    # assert all indices are the same
    for batch, indices in zip(batches, store[HelperData.dl_next_idx_ids]):
        assert torch.LongTensor(batch).equal(
            torch.LongTensor(indices)
        ), "indices are not the same"

    mp_dataloader = DataLoader(
        a,
        batch_size=3,
        num_workers=2,
        persistent_workers=True,
        shuffle=True,
    )

    patch_dataloaders(store)
    store[HelperData.dl_next_idx_ids] = []
    batches = []
    for batch in mp_dataloader:
        batches.append(batch.long())
    assert len(batches) == len(
        store[HelperData.dl_next_idx_ids]
    ), "number of batches is not the same"
    # assert all indices are the same
    for batch, indices in zip(batches, store[HelperData.dl_next_idx_ids]):
        assert torch.LongTensor(batch).equal(
            torch.LongTensor(indices)
        ), "indices are not the same"
    unpatch(store[HelperData.patches])


def test_interrupt_mp():
    store = {HelperData.dl_next_idx_ids: []}
    a = torch.arange(0, 10)
    mp_dataloader = DataLoader(
        a, batch_size=3, num_workers=2, persistent_workers=True, shuffle=True
    )
    patch_dataloaders(store)

    for batch in mp_dataloader:
        ids = store[HelperData.dl_next_idx_ids]
        assert torch.LongTensor(batch).equal(torch.LongTensor(ids.pop(0)))

    sp_dataloader = DataLoader(a, batch_size=3, num_workers=0, shuffle=True)
    for batch in sp_dataloader:
        ids = store[HelperData.dl_next_idx_ids]
        assert torch.LongTensor(batch).equal(torch.LongTensor(ids.pop(0)))
