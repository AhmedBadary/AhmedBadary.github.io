---
layout: NotesPage
title: Ruby
permalink: /work_files/web_dev/Ruby
prevLink: /work_files/web_dev.html
---

<div markdown="1" class = "TOC">
# Table of Contents

  * [Data Types, Input, Output, Constants, File I/O](#content1)
  {: .TOC1}
  * [Conditionals and Looping](#content2)
  {: .TOC2}
  * [Functions and Exceptions](#content3)
  {: .TOC3}
  * [String Methods](#content4)
  {: .TOC4}
  * [Classes, Objects, Inheritance](#content5)
  {: .TOC4}
  * [Modules, Include, Prepend, Polymorphism and Duck Typing](#content6)
  {: .TOC4}
  * [human.rb](#content7)
  {: .TOC4}
  * [smart.rb](#content8)
  {: .TOC4}
  * [Symbols](#content9)
  {: .TOC4}
  * [Arrays](#content10)
  {: .TOC4}
  * [Hashing](#content11)
  {: .TOC4}
  * [Enumerables](#content12)
  {: .TOC4}
  * [File I/O](#content13)
  {: .TOC4}

</div>

***
***

## Data Types, Input, Output, Constants, File I/O
{: #content1}

```ruby
# Ruby is a dynamic, interpreted, object oriented language
# Everything is an object and all of those objects can be overwritten
# Ruby is one of the easiest languages to read and with Rails it may be
# the best web development option period

# print prints the string to screen without a newline
print "Enter a Value: "

# Variables start with a lowercase letter or _ and may contain numbers
# gets stores input from the user and to_i turns it into an integer
first_num = gets.to_i

print "Enter Another Value: "

second_num = gets.to_i

# puts prints output plus a newline, to_s converts the variable into a
# string, you can combine values using +
puts first_num.to_s + " + " + second_num.to_s + " = " +
(first_num + second_num).to_s

# Arithmetic Operators
puts "6 + 4 = " + (6+4).to_s
puts "6 - 4 = " + (6-4).to_s
puts "6 * 4 = " + (6*4).to_s
puts "6 / 4 = " + (6/4).to_s
puts "6 % 4 = " + (6%4).to_s

# Integers or Fixnums can store extremely large numbers
# They can go well beyond 4,611,686,018,427,387,903

numOne = 1.000

# You must put a 0 before your floats
num99 = 0.999

# Floating point calculations tend to be accurate
puts numOne.to_s + " - " + num99.to_s + " = " + (numOne - num99).to_s

# 14 Digits of accuracy is the norm
big_float = 1.12345678901234
puts (big_float + 0.00000000000005).to_s

# Everything is an object
puts 1.class
puts 1.234.class
puts "A String".class

# A Constant starts with an uppercase letter and if you try to change the
# value you will be warned

A_CONSTANT = 3.14
A_CONSTANT = 1.6
puts A_CONSTANT

# Returns a File object for writing
write_handler = File.new("yourSum.out", "w")

# Puts the text in the file
write_handler.puts("Random Text").to_s

# Closes the file
write_handler.close

# Read data from the defined file
data_from_file = File.read("yourSum.out")

puts "Data From File : " + data_from_file

# Use load to execute the code in another Ruby file
load "rubyOrNotRuby2.rb"
```

## Conditionals and Looping
{: #content2}
```ruby
=begin
Multiline comments
=end

# --------------- IF ELSE ELSIF ---------------

age = 12

# If statements perform different actions depending on conditions
# Comparison Operators : == != < > <= >=
# Logical Operators : && || ! (and, or, not)

if (age >= 5) && (age <= 6)
  puts "You're in Kindergarten"
elsif (age >= 7) && (age <= 13)
  puts "You're in Middle School"
  puts "Yeah"
elsif (age >= 14) && (age <= 18)
  puts "You're in High School"
else
  puts "Stay Home"
end

puts "true && false = " + (true && false).to_s
puts "true && false = " + (true || false).to_s
puts "!true = " + (!true).to_s

# Comparison operator that returns 0 (equal), 1 if 1st is greater and
# -1 if 1st is less then
puts "5 <=> 10 = " + (5 <=> 10).to_s

# unless is another comparison operator

unless age > 4
  puts "No School"
else
  puts "Go to School"
end

# You can add condtions to the output that proceeds it

puts "You're Young" if age < 30

# --------------- CASE ---------------

print "Enter Greeting : "

# chomp removes /n from input

greeting = gets.chomp

# Case is used when you have a limited number of options

case greeting
when "French", "french"
  puts "Bonjour"

  # without end it would check the next condition
  exit

when "Spanish", "spanish"
  puts "Hola"
  exit

# else is the default
else
  puts "Hello"
end

# --------------- TERNARY OPERATOR ---------------

age = 50

# Format (condition) ? "Returned if true" : "Returned if false"

puts (age >= 50) ? "Old" : "Young"

# --------------- LOOP ---------------

x = 1

# Loops until you break out of loop

loop do
  # Shortcut for x = x + 1
  x += 1

  # Slips this iteration of the loop
  next unless (x % 2) == 0
  puts x

  # Ends the looping
  break if x >= 10
end

# --------------- WHILE LOOP ---------------
# Do stuff while the condition is met

y = 1

while y <= 10
  y += 1
  next unless (y % 2) == 0
  puts y
  break if y >= 10
end

z = 1

begin
  z += 1
  next unless (z % 2) == 0
  puts z
  break if z >= 10
end while z <= 10

# --------------- UNTIL LOOP ---------------
# Do stuff until a condition is met

a = 1

until a >= 10
  a += 1
  next unless (a % 2) == 0
  puts a
end

# --------------- FOR LOOPS ---------------

# Create an array of numbers

numbers = [1, 2, 3, 4, 5]

# Cycles through every item in numbers temporarily storing them in number
# #{variable} can be used to insert values

for number in numbers
  puts "#{number}, "
end

groceries = ["bananas", "sweet potatoes", "pasta", "tomatoes"]

# Cycles through every item in groceries temporarily storing them in food

groceries.each do |food|
  puts "Get some #{food}"
end

# Cycles through numbers 0 through 5

(0..5).each do |i|
  puts "# #{i}"
end
```

## Functions and Exceptions
{: #content3}

```ruby
# --------------- FUNCTIONS ---------------
 
# Functions start with def, function name, parameters if any
# They can return, or not a value
 
def add_nums(num_1, num_2)
  return num_1.to_i + num_2.to_i
end
 
puts add_nums(3,4)
 
# Variables are passed by value so the'r value can't be changed in a function
 
x = 1
 
def change_x(x)
  x = 4
end
 
change_x(x)
 
puts "x = #{x}"
 
# --------------- EXCEPTIONS ---------------
 
# We catch exceptions with begin and rescue
 
print "Enter a number : "
 
first_num = gets.to_i
 
print "Enter Another : "
 
second_num = gets.to_i
 
begin
  answer = first_num / second_num
 
rescue # You could use rescue ZeroDivisionError
  puts "You can't divide by zero"
  exit
end
 
puts "#{first_num} / #{second_num} = #{answer}"
 
# You can throw your own exceptions with raise
 
def check_age(age)
  raise ArgumentError, "Enter Positive Number" unless age > 0
end
 
# You catch them with rescue
 
begin
  check_age(-1)
rescue ArgumentError
  puts "That is an impossible age"
end
```

## String Methods
{: #content4}

```ruby
# --------------- STRINGS ---------------

# Strings are a series of characters between " or '
# String interpolation doesn't work with '' and neither do backslash
# characters like newline

puts "Add Them #{4 + 5} \n\n"
puts 'Add Them #{4 + 5} \n\n'

# A here-doc is normally used when you want a multiline string

multiline_string = <<EOM
This is a very long string
that contains interpolation
like #{4 + 5} \n\n
EOM

puts multiline_string

# ---------------------------------------

# Find all string methods by typing irb in terminal and "string".methods

first_name = "Derek"

last_name = "Banas"

# You can combine or concatenate strings with +

full_name = first_name + " " + last_name

middle_name = "Justin"

# Combining strings with interpolation

full_name = "#{first_name} #{middle_name} #{last_name}"
puts full_name

# You can check if a string contains a string with include

puts full_name.include?("Justin")

# Get the length of a string

puts full_name.size

# Count the number of vowels

puts "Vowels : " + full_name.count("aeiou").to_s

# Count the consonants

puts "Consonants : " + full_name.count("^aeiou").to_s

# You can check if a string starts with another string

puts full_name.start_with?("Banas")

# Return the index for the match

puts "Index : " + full_name.index("Banas").to_s

# Check equality of strings

puts "a == a : " + ("a" == "a").to_s

# Check if they are the same object

puts "\"a\".equal?(\"a\") : " + ("a".equal?"a").to_s
puts first_name.equal?first_name

# Changing Case

puts full_name.upcase
puts full_name.downcase
puts full_name.swapcase

# Stripping white space

full_name = "       " + full_name

full_name = full_name.lstrip
full_name = full_name.rstrip
full_name = full_name.strip

puts full_name

# Formatting Strings

puts full_name.rjust(20, '.')
puts full_name.ljust(20, '.')
puts full_name.center(20, '.')

# Chop eliminates the last character

puts full_name.chop

# Chomp eliminates \n or a specific string

puts full_name.chomp
puts full_name.chomp('as')

# Delete deletes provided characters

puts full_name.delete("a")

# Split a string into an array

name_array = full_name.split(//)
puts name_array

name_array = full_name.split(/ /)
puts name_array

# String Conversions

puts "a".to_i
puts "2".to_f
puts "2".to_sym

# Escape sequences
# \\  Backslash
# \'  Single-quote
# \"  Double-quote
# \a  Bell
# \b  Backspace
# \f  Formfeed
# \n  Newline
# \r  Carriage
# \t  Tab
# \v  Vertical tab
```

## Classes, Objects, Inheritance
{: #content5}

```ruby
# In an object oriented language we model real world objects using classes
# Every object has attributes (instance variables) and capabilities (methods)

class Animal

  # Called when an Animal object is created
  # You should set default values here
  def initialize
    puts "Creating a New Animal"
  end

  def set_name(new_name)

    # Sets the value for an instance variable
    @name = new_name

  end

  def get_name
    @name
  end

  # Provides another way to get the value for name
  def name
    @name
  end

  # Provides another way to set the value for name
  def name=(new_name)

    # We can eliminate bad input in our setters
    if new_name.is_a?(Numeric)
      puts "Name Can't Be a Number"
    else
      @name = new_name
    end
  end

end

# Creates a new Animal object
cat = Animal.new

# Sets the Animals name
cat.set_name("Peekaboo")

# get_name returns the value of name
puts cat.get_name

# Using the alternative way of getting the name value
puts cat.name

# Using the alternative way of setting a value for name
cat.name = "Sophie"
puts cat.name

class Dog

  # Shortcut for creating all getter functions
  attr_reader :name, :height, :weight

  # Shortcut for creating all setter functions
  attr_writer :name, :height, :weight

  # Creates setter and getter methods (Use this One)
  attr_accessor :name, :height, :weight

  def bark
    return "Generic Bark"
  end
end

rover = Dog.new

rover.name = "Rover"

puts rover.name

puts rover.bark

# When you inherit from another class you get all its methods and variables
# You can only inherit from one class

class GermanShepard < Dog

  # You can overwrite methods as you need

  def bark
    return "Loud Bark"
  end
end

max = GermanShepard.new

max.name = "Max"

printf "%s goes %s \n", max.name, max.bark()
```

## Modules, Include, Prepend, Polymorphism and Duck Typing
{: #content6}

```ruby
# Modules are made up of methods and variables, but they can't be instantiated
# They are used to add functionality to a class

# Allows you access to the Human module (Ruby 2+)
require_relative "human"
require_relative "smart"

module Animal
  def make_sound
    puts "Grrrrrr"
  end
end

# You can inherit a modules methods with prepend or include
# You can inherit from numerous methods instead of one class

class Dog
  include Animal
end

rover = Dog.new
rover.make_sound

class Scientist
  include Human
  prepend Smart # Any methods in Smart will superseed those in the class

  def act_smart
    return "E = mc^2"
  end

end

einstein = Scientist.new

einstein.name = "Albert"

puts einstein.name

einstein.run

puts einstein.name + " says " + einstein.act_smart

# --------------- POLYMORPHISM ---------------

class Bird
  def tweet(bird_type)
    bird_type.tweet
  end
end

class Cardinal < Bird
  def tweet
    puts "Tweet tweet"
  end
end

class Parrot < Bird
  def tweet
    puts "Squawk"
  end
end

# Statically typed languages use duck typing to achieve polymorphism
# Ruby pays less attention to the class type versus the methods that can
# be called on it

generic_bird = Bird.new

generic_bird.tweet(Cardinal.new)
generic_bird.tweet(Parrot.new)
```
## human.rb
{: #content7}

```ruby
module Human
  # Creates setter and getter methods
  attr_accessor :name, :height, :weight

  def run

    # self provides this specific objects value for a variable

    puts self.name + " runs"
  end

end
```

## smart.rb
{: #content8}

```ruby
module Smart

  def act_smart
    return "E = mc²"
  end

end
```

## Symbols
{: #content9}

```ruby
# --------------- SYMBOLS ---------------

# Symbols are strings that can't be changed
# You use them to conserve memory and to speed string comparisons
# A symbol is a way to pass string data if:
# The string value won't change
# The string doesn't need access to string methods

:derek

puts :derek
puts :derek.to_s
puts :derek.class
puts :derek.object_id

# Many core Ruby methods take symbols as arguments such as
# attr_accessor :name, :height, :weight that we used earlier

# Symbols are also used as keys for hashes
```

## Arrays
{: #content10}

```ruby
# --------------- ARRAYS ---------------

# Creating arrays

array_1 = Array.new
array_2 = Array.new(5) # Gets nil as default
array_3 = Array.new(5, "empty")
array_4 = [1,"two",3,5.5] # You can store multiple object types

puts array_1
puts array_2
puts array_3
puts array_4

# Indexes start at 0
puts array_4[2]

# Return 2 values starting at the 2nd index
puts array_4[2,2].join(", ")

# Return values in index 0, 1 and 3
puts array_4.values_at(0,1,3).join(", ")

# Add 0 at the beginning of the array
array_4.unshift(0)
puts array_4.join(", ")

# Remove the first item in the array
array_4.shift()
puts array_4.join(", ")

# Add 100 and 200 to the end of the array
array_4.push(100, 200)
puts array_4.join(", ")

# Remove item at the end of the array
array_4.pop
puts array_4.join(", ")

# Add one array to the end of another
array_4.concat([10,20,30])
puts array_4.join(", ")

# Array Methods

puts "Array Size : " + array_4.size().to_s
puts "Array Contains 100 : " + array_4.include?(100).to_s
puts "How Many 100s : " + array_4.count(100).to_s
puts "Array Empty : " + array_4.empty?.to_s

# Convert an array into a string
puts array_4.join(", ")

# Print and Inspect the array
p array_4

# Output array in loop
array_4.each do |value|
  puts value
end
```


## Hashes
{: #content11}

```ruby

# --------------- HASHS ---------------

# A hash is a collection of key object pairs

number_hash = { "PI" => 3.14,
                "Golden" => 1.618,
                "e" => 2.718}

puts number_hash["PI"]

superheroes = Hash["Clark Kent", "Superman", "Bruce Wayne", "Batman"]

puts superheroes["Clark Kent"]

# Add to a hash
superheroes["Barry Allen"] = "Flash"
puts superheroes["Barry Allen"]

# Set a default key value
samp_hash = Hash.new("No Such Key")
puts samp_hash["Dog"]

superheroines = Hash["Lisa Morel", "Aquagirl", "Betty Kane", "Batgirl"]

# Combine 2 hashes in a destructive way with update or with no overwriting
# with merge

superheroes.update(superheroines)

# Print out each key and value

superheroes.each do |key, value|
  puts key.to_s + ' : ' + value
end

puts "Has Key Lisa Morel : " + superheroes.has_key?("Lisa Morel").to_s
puts "Has Value Batman : " + superheroes.has_value?("Batman").to_s
puts "Is Hash Empty : " + superheroes.empty?.to_s
puts "Size of Hash : " + superheroes.size.to_s

# Delete a key value

superheroes.delete("Barry Allen")

puts "Size of Hash : " + superheroes.size.to_s
```



## Enumerable 
{: #content12}

```ruby
# Classes that include the Enumerable module gain collection capabilities
# but they must define a function called each

class Menu
  include Enumerable

  # Each provides items one at a time

  def each
    yield "pizza"
    yield "spaghetti"
    yield "salad"
    yield "bread"
    yield "water"
  end
end

menu_options = Menu.new

# We can cycle through all the options

menu_options.each do |item|
  puts "Would you like : #{item}"
end

# Check to see if we have pizza
p menu_options.find {|item| item == "pizza"}

# Return items 5 letters in length
p menu_options.select {|item| item.size <= 5}

# Reject items that meet the criteria
p menu_options.reject {|item| item.size <= 5}

# Return the first item
p menu_options.first

# Return the first 2
p menu_options.take(2)

# Return the everything except the first 2
p menu_options.drop(2)

# Return the minimum item
p menu_options.min

# Return the maximum item
p menu_options.max

# Sort the items
p menu_options.sort

# Return each item in reverse order
menu_options.reverse_each {|item| puts item}
```


## File I/O
{: #content13}

```ruby
# The class File provides for file manipulation

# Create a file for writing
file = File.new("authors.out", "w")

# Add lines
file.puts "William Shakespeare"
file.puts "Agatha Christie"
file.puts "Barbara Cartland"

# Close file
file.close

# Output everything in the file
puts File.read("authors.out")

# Open file for appending
file = File.new("authors.out", "a")
file.puts "Danielle Steel"
file.close
puts File.read("authors.out")

# Create another file containing data separated by commas
file = File.new("author_info.out", "w")
file.puts "William Shakespeare,English,plays and poetry,4 billion"
file.puts "Agatha Christie,English,who done its,4 billion"
file.puts "Barbara Cartland,English,romance novels,1 billion"
file.puts "Danielle Steel,English,romance novels,800 million"
file.close

# Cycle through the data to write a sentence
File.open("author_info.out") do |record|
  record.each do |item|

    # Split each line into 4 parts based on commas
    name, lang, specialty, sales = item.chomp.split(',')
    puts "#{name} was an #{lang} author that specialized in #{specialty}. They sold over #{sales} books."
  end
end
```