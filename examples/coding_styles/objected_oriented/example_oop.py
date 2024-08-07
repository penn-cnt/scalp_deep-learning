class Animal:
   
   def __init__(self, name):
      self.name = name
   
   def speak(self):
      raise NotImplementedError("Subclass must implement this method.")
   
   def move(self):
      raise NotImplementedError("Subclass must implement this method.")
   
   def perform_trick(self):
      return f"{self.name} performed a trick!"
   
   def sleep(self):
      return f"{self.name} is taking a nap."

class Dog(Animal):
   def speak(self):
      return "Woof!"
   
   def move(self):
      return f"{self.name} Runs"

class Cat(Animal):
   def speak(self):
      return "Meow!"
   
   def move(self):
      return f"{self.name} Jumps"

class Shark(Animal):
   def speak(self):
      return "Blub"
   
   def move(self):
      return f"{self.name} Swims"

if __name__ == '__main__':

   # Dog stuff
   print("\nOur dog friend:")
   dog = Dog("Buddy")
   print(dog.name)      
   print(dog.speak())
   print(dog.move())
   print(dog.perform_trick())
   print(dog.sleep())

   # Cat stuff
   print("\nOur cat friend:")
   cat = Cat("Mittens")
   print(cat.name)      
   print(cat.speak())
   print(cat.move())
   print(cat.perform_trick())
   print(cat.sleep())

   # Shark stuff
   print("\nOur shark friend:")
   shark = Shark("Larry")
   print(shark.name)      
   print(shark.speak())
   print(shark.move())
   print(shark.perform_trick())
   print(shark.sleep())