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

### 2. Objects and Instances

#### 2.1 Sorting list of Instances
1. Using sorted function and specify key as the state of the instance to comapare on.
2. Define a method named $__lt__$ which stands for “less than”. It takes two instances as arguments: self and an argument for another instance. It returns True if the self instance should come before the other instance, and False otherwise. Python translates the expression a < b into $a.__lt__(b)$. When we call sorted(L) without specifying a value for the key parameter, it will sort the items in the list using the $__lt__$ method defined for the class of items.

#### 2.2 Class variables and instance variables

Instance variables: Every instance of a class can have its own instance variables. These variables represent the properties or attributes of a specific instance.  

Class variables: Classes can also have their own class variables. These variables are shared among all instances of the class and belong to the class itself. They are defined outside of any method in the class. Class methods are also a form of class variables. 

- Search order for variables: When accessing a variable using the instance name, Python first searches inside the instance for the variable. If it is not found, it then searches inside the class. If the variable is still not found, a runtime error occurs.  

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