from budgetbase import budget

class constantBudget(budget):
    def __init__(self, roundBudget = 0):
        super().__init__("Constant Budget")
        self.__budget = totalBudget

    def getTotalBudget(self):
        return self.__budget

    def getTotalBudgetAt(self, rounds):
        return self.__budget * rounds

    def getBudget(self):
        return self.__budget

    def getBudgetAt(self, index):
        return self.__budget
