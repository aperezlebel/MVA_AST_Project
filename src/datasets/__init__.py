from .BTCDataset import BTCDataset
from .EquityDataset import EquityDataset


available_datasets = {
    'btc': BTCDataset,
    'ety': EquityDataset,
}
