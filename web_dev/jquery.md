---
layout: NotesPage
title: JQuery
permalink: /work_files/web_dev/jquery
---

<div markdown="1" class = "TOC">
# Table of Contents

  * [Basic Interaction, Selection, and Modification](#content1)
  {: .TOC1}
  * [DOM Events and Event Listeners](#content2)
  {: .TOC2}
  * [Form Processing](#content3)
  {: .TOC3}
</div>



***
***

## Basic Interaction, Selection, and Modification 
{: #content1}
1. **Including jQuery:**{: style="color: SteelBlue  "}{: .bodyContents1 #bodyContents11}
: ```<script src="https://ajax.googleapis.com/ajax/libs/jquery/2.1.4/jquery.min.js"></script>```
2. **Main Function:**{: style="color: SteelBlue  "}{: .bodyContents1 #bodyContents12}
: ```$();```
3. **Selectors:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents13}[^1]
    * **By TagName:**
    :   ```$("TagName");```
    > e.g. \\
    ```$("h1");```
    * **By ID:**
    :   ```$("#ID");```
    * **By Class:**
    :   ```$(".class");```
4. **jQuery Collections and DOM Nodes:**{: style="color: SteelBlue"}{: .bodyContents1  #bodyContents14}
    * **To turn a jQuery Collection into a DOM Node:**
    :   ```javascript
        var jQCollection = $("#head");
        var DomNode = jQCollection[0];
        ```
    * **To turn a DOM Node into a jQuery Collection:**
    :   ```javascript
        var jQCollection = $(DomNode);
        ```
4. **Modify Elements Content:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents15}
```javascript
$("Selector").text("Hello, World!");
```
> e.g. \\
``` $("h1").text(Hello, World!);```\\
``` $("p").html(<span class = "hello">Hello, World!</span>);```
5. **Modify CSS Attributes:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents16}
```javascript
$("Selector").css("CSS-Attribute", "Value");
```
> e.g. \\
``` $("p").css("color", "green");```
6. **Modify HTML Attributes:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents17}
```javascript
$("Selector").attr("HTML-Attribute", "Value");
```
> e.g. \\
``` $("a").attr("href", "https://en.wikipedia.org/wiki/Cats");```

7. **Add Class Name Attribute:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents18}
```javascript
$("Selector").addClass("className");
```
> e.g. \\
``` $("h2").addClass("header2");```
8. **Creating DOM Elements:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents19}
```javascript
$("<TagName>");
```
> e.g. \\
``` var paragraph = $("<p>");```
    * **Adding Created Element to the DOM:**{: style="color: SteelBlue"}
    ```javascript
    // Appending
    $("Selector").append(variableName);
    // Prepending. Can, also, use .after()
    $("Selector").prepend(variableName);
    ```
        > e.g. [^2]
        ```javascript
        var paragraph = $("<p>");
        $("body").append(paragraph);
        ```

        > Another variation without saving the variable name
        ```javascript
        $("<p>").appendTo(".className");
        ```
9. **Looping through Elements / for loops:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents10} \\
The following are equivalent
> for loop
```javascript
        // for loop
        var paragraphs = $('p');
        for (var i = 0; i < paragraphs.length; i++) {
            var element = paragraphs[i]; // DOM node
            var paragraph = $(element);
            paragraph.html( "Hello, World!");
        }
```
> each loop
```javascript
        // each()
        paragraphs.each(function(index, this) {
            var paragraph = $(this);
            paragraph.html( "Hello, World!");
        });
```
10. **Chaining:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents11}
> This
```javascript
var newP = $("<p>"); // Create new Paragraph Element
newP.text("The crocodiles have eaten this ENTIRE PAGE!"); // Add text to the created Paragraph
newP.addClass("crocodile"); // Add a class to Paragraph
$("body").append(newP); // Append Paragraph to the body
```
> is Equivalent to this
```javascript
$("<p>").text("The crocodiles have eaten this ENTIRE PAGE!").addClass("crocodile").appendTo("body");
```

    This is possible because most *jQuery* functions return *jQuery Collections*.




[^1]: Returns a jQuery Collection, NOT a DOM Node.
[^2]: Appends paragraph to the body-tag, making it the last tag in the body.
[^3]: Open me for Examples.
***

## DOM Events and Event Listeners
{: #content2}
1. **Adding Event Listeners:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents21}
```javascript
    ("#button").on("click", function(event) {  // Select the Button
        console.log("you clicked me!!");  // Specify the Event 
        });
```
2. **Triggering events (Manually):**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents22} \\
`$("#save-button").trigger("click");`
3. **Types of Events:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents23}
    * **mouse events**: 'click' , 'mousedown'/'mouseup', 'mousemove', 'mouseenter'/'mouseleave'
    * **keyboard events**: 'keydown', 'keypress', 'keyup'
    * **touch events**: 'touchdown', 'touchup', 'touchstart', ‘touchcancel’
    * **drag events**: 'dragstart', 'dragend' (Many developers use jQueryUI for drag functionality, as it can be tricky to use the drag events directly.)
    * **form events**: 'submit', 'change', 'focus'/'blur'
    * **window events**: 'scroll', 'reload', 'hashchange'
> source \\
> [Khan Academy](https://www.khanacademy.org/computing/computer-programming/html-js-jquery/dom-events-with-jquery/a/dom-events-and-properties)[^3]
4. **Checking DOM Readiness:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents24}
```javascript
$(document).ready(function() {
    someFunction();
});
```

***

## Form Processing
{: #content3}
1. **Types of Events:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents31}
```javascript
$("form").on("submit", function() {
    // process form     
});
```
2. **Preventing Webpages' Default Actions:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents32}
```javascript
$("form").on("submit", function(event) {
   event.preventDefault();
   // process form
});
```
3. **Retrieveing Submitted Information:**{: style="color: SteelBlue"}{: .bodyContents3 #bodyContents33}
```javascript
$("form").on("submit", function() {
    // Find the input with name='age' 
    var age = $(this).find('[name=age]');
    // Store the value of the input with name='age'
    var ageValue = age.val();
});
```
