# import os

# print(os.path.dirname(__file__))

class Grandparent():
    def __init__(self):
        self.show()
        
    def show(self):
        print("Grandparent")

class Parent(Grandparent):
    def __init__(self):
        super().__init__()
        self.value = "Parent"
    
    def show(self):
        print("Parent")
    
class Child(Parent):
    def __init__(self):
        self.value = "Child"
        super().__init__()

    def show(self):
        print("Child")

# obj1 = Parent()
obj2 = Child()

# obj1.show()
# obj2.show()