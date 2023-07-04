from .budgetbase import budget

class ConstantBudget(budget):
    def __init__(self, roundBudget = 0):
        super().__init__("Constant Budget")
        self.__budget = roundBudget

    def getTotalBudget(self):
        return self.__budget

    def getTotalBudgetAt(self, rounds):
        return self.__budget * rounds

    def getBudget(self):
        return self.__budget

    def getBudgetAt(self, index):
        return self.__budget
