from budgetbase import budget

class arrayBudget(budget):
    def __init__(self, budgetArray):
        super().__init__("Array Budget")
        self.__budgets = budgetArray

    def getTotalBudget(self):
        return sum(self.__budgets)

    def getTotalBudgetAt(self, rounds):
        return sum(self.__budgets[:min(index, len(self.__budgets) - 1)]) +
            self.__budgets[-1] * max(0, index - len(self.__budgets) + 1)

    #This method can't be implemented since we need an index to know how many rounds have past.
    def getBudget(self):
        return None

    def getBudgetAt(self, index):
        return self.__budgets(min(index, len(self.__budgets)-1))
