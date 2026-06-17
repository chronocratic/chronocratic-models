__all__ = ["RecurrentCellType"]

from enum import StrEnum


class RecurrentCellType(StrEnum):
    LSTM = "lstm"
    GRU = "gru"
    RNN = "rnn"
