### 1. Constructing Classes

#### 1.1 User Defined Classes

Class definitions can appear anywhere in a program, but they are usually near the beginning (after the import statements). Class name is typically capitalized.  

Every class should have a method with the special name $__init__$. This initializer method, often referred to as the **constructor**, is automatically called whenever a new instance of Point is created. It gives the programmer the opportunity to **set up the attributes** required within the new instance by giving them their initial state values.

```Python
class Point:
    """ Point class for representing and manipulating x,y coordinates. """

    def __init__(self):

        self.x = 0
        self.y = 0

p = Point()         # Instantiate an object of type Point
q = Point()         # and make a second point
}
```

**Instantiation**: It may be helpful to think of a class as a factory for making objects. The class itself isn’t an instance of a point, but it contains the machinery to make point instances. Every time you call the constructor, you’re asking the factory to make you a new object. 

p and q are each bound to different Point instances. Even though both have x and y instance variables set to 0, they are different objects. Thus p is q evaluates to False.

#### 1.2 Adding parameters to the constructor

We can make our class constructor more generally usable by putting extra parameters into the $__init__$ method, as shown in this example.

```Python
class Point:
    """ Point class for representing and manipulating x,y coordinates. """

    def __init__(self, initX, initY):

        self.x = initX
        self.y = initY

p = Point(7,6)
```
This is a common thing to do in the $__init__$ method for a class: take in some parameters and save them as **instance variables**.  

The $__init__$ method does more than just save parameters as instance variables. For example, it might parse the contents of those variables and do some computation on them, storing the results in instance variables. It might even make an Internet connection, download some content, and store that in instance variables.

#### 1.4 Adding other methods to the class

A method behaves like a function but it is invoked on a specific instance. Methods are accessed using dot notation. 

All methods defined in a class that operate on objects of that class will have self as their first parameter. Again, this serves as a reference to the object itself which in turn gives access to the state data inside the object.

- special underscore methods
  
![m2_1](src/m2_1.png)


#### 1.5 Public and private instance variables

- Idea of abstraction: Others should not need to know exactly how you implemented your class; they should just be able to call the methods you have defined.

- How: Python does not “enforce” the idea of private instance variables. Using underscores in instance variable names (e.g., _iv or __iv) is a way to signal that an instance variable should be private.

- Python “mangles” any instance variables that start with two underscores to make them more difficult to access, by adding the name of the class to the beginning of the instance variable name (_classname__variablename). For example, __age becomes _Person__age. 


#### 1.6 Comparison of functions and methods
Python functions and methods are similar in many ways. They are both callable objects that can be invoked with arguments and return a value.  
The main difference between functions and methods is that **methods are defined within a class and are associated with instances of that class**. Methods are often considered "selfish" because they always put their self (the instance they belong to) first.  
Functions, on the other hand, live outside of any individual instance or object.  

#### 1.7 Testing Classes

To test a user-defined class, you will create test cases that check whether instances are created properly, and you will **create test cases for each of the methods** as functions, by invoking them on particular instances and seeing whether they produce the correct return values and side effects.

There are two types of tests depending on what the methods try to do:  
1. Return value test: whether they produce the correct return values
2. Side effect test: To test a method that changes the value of an instance variable

### 2. Objects and Instances

#### 2.1 Sorting list of Instances
1. Using sorted function and specify key as the state of the instance to comapare on.
2. Define a method named $__lt__$ which stands for “less than”. It takes two instances as arguments: self and an argument for another instance. It returns True if the self instance should come before the other instance, and False otherwise. Python translates the expression a < b into $a.__lt__(b)$. When we call sorted(L) without specifying a value for the key parameter, it will sort the items in the list using the $__lt__$ method defined for the class of items.

#### 2.2 Class variables and instance variables

Instance variables: Every instance of a class can have its own instance variables. These variables represent the properties or attributes of a specific instance.  

Class variables: Classes can also have their own class variables. These variables are shared among all instances of the class and belong to the class itself. They are **defined outside of any method in the class**. Class methods are also a form of class variables. 

- When the interpreter sees an expression of the form obj.varname, it:
  1) Checks if the object has an instance variable set. If so, it uses that value.
  2) If it doesn’t find an instance variable, it checks whether the class has a class variable. If so it uses that value.
  3) If it doesn’t find an instance or a class variable, it creates a runtime error.

```Python
class MyClass:
    def __init__(self):
        self.f = lambda: 20
    def f(self):
        return 30

inst_1 = MyClass(5)
inst_1.f() # 20
```
```Python
class MyClass:
    class_lst = [1, 2]
    def __init__(self):
        self.instance_lst = [3, 4, 5]
        
inst_1 = MyClass()
inst_2 = MyClass()

print(inst_1.instance_lst is inst_2.instance_lst,
      inst_1.class_lst is inst_2.class_lst)
# False, True
```

### 3. Inheritance

Classes can “inherit” methods and class variables from other classes. It’s useful when someone else has defined a class in a module or library, and you just want to override a few things without having to reimplement everything they’ve done.
#### 3.1 Inheriting Varibales and Methods

In the definition of the inherited class, you only need to specify the methods and instance variables that are different from the parent class (the **parent class**, or the **superclass**).

```Python
# Here's the new definition of class Cat, a subclass of Pet.
class Cat(Pet): # the class name that the new class inherits from goes in the parentheses, like so.
    sounds = ['Meow'] # existing class variables can be updated 

    def chasing_rats(self):
        return "What are you doing, Pinky? Taking over the world?!"

```

This is how the interpreter looks up attributes:

1. First, it checks for an instance variable or an instance method by the name it’s looking for.

2. If an instance variable or method by that name is not found, it checks for a class variable.

3. If no class variable is found, it looks for a class variable in the parent class.

4. If no class variable is found, the interpreter looks for a class variable in THAT class’s parent (the “grandparent” class).

5. This process goes on until the last ancestor is reached, at which point Python will signal an error.

If a method is defined for a class, and also defined for its parent class, the subclass’ method is called and not the parent’s. This follows from the rules for looking up attributes.

#### 3.2 Invoking Parent Class's Method

Sometimes the parent class has a useful method, but you just need to execute a little extra code when running the subclass’s method. You can override the parent class’s method in the subclass’s method with the same name, but also invoke the parent class’s method with **super().method()**. 

```Python
class Dog(Pet):
    sounds = ['Woof', 'Ruff']

    def feed(self):
        # equivalent: Pet.feed(self)
        super().feed()
        print("Arf! Thanks!")

d1 = Dog("Astro")

d1.feed()

```

Pet.feed(self) is equivalent to super().feed(). However it is less flexible in cases when parent classes change names or the hierarchy.

#### 3.3 Multiple inheritance

In Python, a class can inherit from more than one parent class. This is called multiple inheritance.  

 It’s generally a good rule to avoid multiple inheritance unless it provides a clear and significant benefit. Always consider simpler alternatives, such as composition (using an instance of one class as an instance variable inside of another class) or single inheritance, before turning to multiple inheritance.

 ### 4. Decorators
 #### 4.1 Function Wrapping and Decorators

Python has a “decorator” syntax” that allows us to modify the behavior of functions or classes. 

A decorator is a function that **accepts a function as an argument and returns a new function**. The new function is usually a “wrapped” version of the original function. 

The decorator syntax is to place an @ symbol followed by the name of the decorator function on the line before the function definition. 

#### 4.2 Class decorators
There are two ways we can use decorators with classes: (1) by decorating individual class methods or (2) by decorating the class itself.

```Python
def addLogging(func): # The argument, func is a method of a class

    def wrapper(self, x): # x is the argument that we're going to pass to func
        print(f"About to call the method with argument {x}")
        result = func(self, x) # actually call the method and store the result. Need to pass self as the 1st argument as it's required when calling the method 
        print(f"Done with the method invocation with argument {x} on instance {self}. Result: {result}")
        return result # return whatever our function returned

    return wrapper # return our new wrapped function

def addBeep(cls):
    cls.beep = lambda self: print(f"{self.model} says 'Beep!'")
    return cls

@addBeep # decorating the class itself
class Car:
    def __init__(self, make, model, color, mileage):
        self.make = make
        self.model = model
        self.color = color
        self.mileage = mileage

    @addLogging # decorating class methods
    # equivalent to: drive = addLogging(drive)
    def drive(self, miles):
        self.mileage += miles
        return self.mileage

    @addLogging
    def rePaint(self, color):
        self.color = color

    def __str__(self):
        return(f"***{self.color} {self.make} {self.model} with {self.mileage} miles***")

corvette = Car("Chevrolet", "Corvette", "red", 0)

corvette.drive(100)
corvette.beep()

```

#### 4.3 Bulit-in Method Decorators

**@classmethod**: 
- used to define a method that operates on the class itself ( it takes in class as an argument) rather than an instance of the class (as we normally do with methods). This means you can use the class and its properties inside that method rather than a particular instance.
- callable without instantiating the class, but its definition follows Sub class, not Parent class, via inheritance, can be overridden by subclass. That’s because the first argument for @classmethod function must always be cls (class).  
- Often used as a factory method to handle preprocessing when creating an instance.

**@staticmethod**: 
- the static method decorator doesn't need to reference anything about any particular instance or about the class, although it belongs to the class. 
- It is nothing more than a function defined inside a class. It is callable without instantiating the class first. It’s definition is immutable via inheritance.


```Python
class Date(object):
    
    def __init__(self, day=0, month=0, year=0):
        self.day = day
        self.month = month
        self.year = year

    @classmethod
    def from_string(cls, date_as_string):
        day, month, year = map(int, date_as_string.split('-'))
        date1 = cls(day, month, year)
        return date1

    @staticmethod
    def is_date_valid(date_as_string):
        day, month, year = map(int, date_as_string.split('-'))
        return day <= 31 and month <= 12 and year <= 3999

date2 = Date.from_string('11-09-2012')
is_date = Date.is_date_valid('11-09-2012')

```

Compared to implementing the string parsing as a single function elsewhere, classmethod allows you to encapsulate it.  


#### 4.4 Built-in Property Decorators

**@property decorator**: decorate the getter and setter functions in a class in order to have a more natural and controlled way of getting and setting the class variables. 

```Python
class Circle:
    def __init__(self, radius): 
        if radius<0:
            raise ValueError("Radius must be non-negative")
        self.__r=radius
    @property
    def radius(self):
        print("Calling getter")
        return self.__r
    
    @radius.setter
    def radius(self, radius):
        print("Calling setter")
        if radius<0:
            raise ValueError("Radius must be non-negative")
        self.__r=radius

c = Circle(10)
print(c.radius)
c.radius = -5
print(c.radius)

```

### 5. Advanced Functions

#### 5.1 Adding documentation to functions using Docstrings

```Python
def add(a,b):
    """Adds two numbers together"""

class Person:
    """A class to represent a person"""

# 3 ways of checking docstrings of a function/class
print(Person.__doc__)
help(Person)
Person? # Jupyter notebook query only
```

#### 5.2 Dynamic positional arguments with *args

- *args is a special syntax in Python that allows a function to accept a variable number of positional arguments. It is used when we want to pass any number of arguments to a function without specifying the exact number of arguments in advance.
- When defining a function, we can use *args as a parameter. This tells Python that the function can accept any number of positional arguments.
- Inside the function, *args is treated as a tuple that contains all the positional arguments passed to the function. We can then iterate over the *args tuple or perform any other operations on it.

```Python
def add(*args):
    """Adds numbers together"""
    for num in args:
        result+=num
    return result

add(1,2,3,4)
# Alternatively unpack a list of elements as separate arguments with *
L = [1,2,3,4]
add(*L)

```
#### 5.3 Dynamic keyword arguments with **kwargs

- When defining a function, we can use **kwargs as a parameter. This tells Python that the function can accept any number of keyword arguments. 
- Inside the function, *kwargs is treated as a dictionary where the keys are the argument names and the values are the corresponding values passed in.

```Python
def create_user_profile(**kwargs):
    print(kwargs)
    user_profile={}
    for key, value in kwargs.items():
        user_profile[key]=value
    return user_profile

D = {'name':'Alice', 'age':30, 'city':'Wonderland'}
# Alternatively unpack a dictionary of key-value pairs as separate arguments with **
create_user_profile(**D)

```

- One common use case of combining *args and **kwargs is in decorators so that the decorator can work with functions that have any number of arguments and keyword arguments.

```Python
def debug(func):
    def wrapper(*args, **kwargs):
        print("args", args)
        print("kwargs", kwargs)
        return func(*args, **kwargs)
    return wrapper
@debug
def add(a,b,c):
    return a+b+c

add(1,2,3) 
# output: args (1,2,3), no kwargs

```

#### 5.4 Decorators with arguments

By defining a decorator within a larger function, we can pass arguments to customize the behavior of the decorator. 
```Python
def repeat(n): # wrapper function for the decorator to accept additional arguments
    def decorator(func):
        def wrapper(*args,**kwargs):
            for _ in range(n):
                func(*args,**kwargs)
            return func
        return wrapper
    return decorator

@repeat(2) # pass in arguments (cannot pass in func here)
def greet(name):
    print(f"Hello,{name}")
greet("Steve")
print(greet)

```

#### 5.5 Built-in decorators in the functools module

Using decorators can cause us to lose important information about the function, such as its name and docstring. To preserve this metadata, we can use the **@functools.wraps** decorator.

```Python
import functools

def debug(func):
    @functools.wraps(func)
    def wraooer(*args, **kwargs):
        print(f"Calling {func} with args={args} and kwargs={kwargs}")
        return func(*args, **kwargs)
    return wrapper

@debug
def add(a,b):
    """Adds two numbers together"""
    return a+b
print(add.__name__) # output:add 
print(add.__doc__) # output: Adds two numbers together

```

**@functools.cache** helps to remember the return value of a function when called with a specific argument.

```Python
import time
@functools.cache
def factorial(n):
    print(f"Calculating {n}!")
    if n==0:
        return 1
    return n*factorial(n-1)

print(factorial(5))
print(factorial(6)) # will only need to compute 6! thanks to cache
```

### 6. Exceptions

In Python, there are three types of errors:

1) Syntax Errors: These errors occur when the code violates the rules of the Python language. They are usually caused by typos, missing parentheses, or incorrect indentation. Syntax errors prevent the code from running at all.

2) Runtime Errors: Also known as exceptions, these errors occur during the execution of the code. They can be caused by various factors such as dividing by zero, accessing an index out of range, or calling a function that doesn't exist. *Runtime errors can be handled using try-except blocks*.

3) Logical/Semantic Errors: These errors occur when the code runs without any syntax or runtime errors, but it produces incorrect results. Logical errors are usually caused by mistakes in the algorithm or the logic of the code. Debugging techniques like print statements or using a debugger can help identify and fix logical errors.

#### 6.1 Exception handling flow of control

The try/except control structure provides a way to process a run-time error and continue on with program execution.

The exception code can access a variable that contains information about exactly what the error was. 
```Python
try:
    items = ['a', 'b']
    third = items[2]
    print("This won't print")
except Exception as e:
    print("got an error")
    print(e)

print("continuing")

# Output:
# got an error
# list index out of range
# continuing
```

The reason to use try/except is when you have a code block to execute that will sometimes run correctly and sometimes not, depending on conditions you can’t foresee at the time you’re writing the code. In general, when using a try-except, it's better to be as specific as possible.

#### 6.2 Standard Exceptions


All exceptions are objects. The classes that define the objects are organized in a hierarchy, which is shown below. This is important because the parent class of a set of related exceptions will catch all exception messages for itself and its child exceptions. For example, an ArithmeticError exception will catch itself and all FloatingPointError, OverflowError, and ZeroDivisionError exceptions.

```Python
BaseException
 +-- SystemExit
 +-- KeyboardInterrupt
 +-- GeneratorExit
 +-- Exception
      +-- StopIteration
      +-- StopAsyncIteration
      +-- ArithmeticError
      |    +-- FloatingPointError
      |    +-- OverflowError
      |    +-- ZeroDivisionError
      +-- AssertionError
      +-- AttributeError
      +-- BufferError
      +-- EOFError
      +-- ImportError
      +-- LookupError
      |    +-- IndexError
      |    +-- KeyError
      +-- MemoryError
      +-- NameError
      |    +-- UnboundLocalError
      +-- OSError
      |    +-- BlockingIOError
      |    +-- ChildProcessError
      |    +-- ConnectionError
      |    |    +-- BrokenPipeError
      |    |    +-- ConnectionAbortedError
      |    |    +-- ConnectionRefusedError
      |    |    +-- ConnectionResetError
      |    +-- FileExistsError
      |    +-- FileNotFoundError
      |    +-- InterruptedError
      |    +-- IsADirectoryError
      |    +-- NotADirectoryError
      |    +-- PermissionError
      |    +-- ProcessLookupError
      |    +-- TimeoutError
      +-- ReferenceError
      +-- RuntimeError
      |    +-- NotImplementedError
      |    +-- RecursionError
      +-- SyntaxError
      |    +-- IndentationError
      |         +-- TabError
      +-- SystemError
      +-- TypeError
      +-- ValueError
      |    +-- UnicodeError
      |         +-- UnicodeDecodeError
      |         +-- UnicodeEncodeError
      |         +-- UnicodeTranslateError
      +-- Warning
           +-- DeprecationWarning
           +-- PendingDeprecationWarning
           +-- RuntimeWarning
           +-- SyntaxWarning
           +-- UserWarning
           +-- FutureWarning
           +-- ImportWarning
           +-- UnicodeWarning
           +-- BytesWarning
           +-- ResourceWarning
```

### 7. Way of the Programmer
#### 7.1 Debugging with Break Points

The debugger tool in IDEs like VS Code and PyCharm enables you to set breakpoints and navigate through your python code step by step. 

#### 7.2 Django as an example

We regular Python programmers can write a short amount of code to inherit and customize classes from open-source libraries instead of building the wheels from scratch.

