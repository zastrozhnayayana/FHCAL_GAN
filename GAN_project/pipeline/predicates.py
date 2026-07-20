"""
Предикаты были созданы для удобного контроля эпох, в которые вычисляются метрики, во время обучения.
Главный класс TrainPredicate представляет собой просто обёртку над функцией-предикатом.
"""
from abc import abstractmethod
# abc - модуль для создания абстрактных базовых классов. Он позволяет создавать абстрактные методы, которые должны быть реализованы в подклассах.


class TrainPredicate:
    @abstractmethod # каждый подкласс должен реализовать метод __call__
    def __call__(self, *args, **kwargs) -> bool:
        pass

    # в одинарных кавычках, потому что TrainPredicate ещё не до конца определён на момент объявления метода __and__
    def __and__(self, pred2: 'TrainPredicate') -> 'TrainPredicate':
        return AndPredicate(self, pred2)


# sample predicates (примеры предикатов)
class IgnoreFirstNEpochsPredicate(TrainPredicate):
    def __init__(self, n: int):
        self.n = n

    def __call__(self, epoch: int, *args, **kwargs) -> bool:
        return epoch > self.n

# USED
class EachNthEpochPredicate(TrainPredicate):
    def __init__(self, n: int):
        self.n = n

    def __call__(self, epoch: int, *args, **kwargs) -> bool:
        return epoch % self.n == 0

# USED
class AndPredicate(TrainPredicate):
    def __init__(self, pred1: TrainPredicate, pred2: TrainPredicate):
        self.pred1 = pred1
        self.pred2 = pred2

    def __call__(self, *args, **kwargs) -> bool:
        return self.pred1(*args, **kwargs) and self.pred2(*args, **kwargs)
