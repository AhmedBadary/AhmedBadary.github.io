---
layout: NotesPage
title: JQuery
permalink: /work_files/web_dev/jquery
prevLink: /work_files/web_dev.html
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
    * <button>Click for Video</button>{: value="show" src="https://www.youtubeeducation.com/embed/KDFtN6C5mO0" .video_buttons #video_buttons13 onclick="iframePopInject(event);"}
        <div markdown="1"> </div>

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
    * <button>Click for Video</button>{: value="show" src="https://www.youtubeeducation.com/embed/mboB93fon3w" .video_buttons #video_buttons14 onclick="iframePopInject(event);"}
        <div markdown="1"> </div>


5. **Getting info on elements with jQuery:**{: style="color: SteelBlue"}{: .bodyContents1  #bodyContents155}
    * **Text:**
    :   ```javascript
        var selectorText = $("Selector").text();
        ```
    * **HTML:**
    :   ```javascript
        var selectorHTML = $("Selector").html();
        ```
    * **HTML ATTRIBUTE:**
    :   ```javascript
        var selectorHTML = $("Selector").attr("Attribute");
        ```
    * **CSS ATTRIBUTE:**
    :   ```javascript
        var selectorHTML = $("Selector").css("Attribute");
        ```
    * <button>Click for Video</button>{: value="show" src="https://www.youtubeeducation.com/embed/nf5lCuLCpwQ" .video_buttons #video_buttons14 onclick="iframePopInject(event);"}
        <div markdown="1"> </div>


5. **Modify Elements Content:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents15}
    ```javascript
    $("Selector").text("Hello, World!");
    ```
    > e.g. \\
    ``` $("h1").text(Hello, World!);```\\
    ``` $("p").html(<span class = "hello">Hello, World!</span>);```  

    <button>Click for Video</button>{: .video_buttons value="show"
    src="https://www.youtubeeducation.com/embed/kD7uk8RNmjM" onclick="iframePopInject(event);"}
        <span markdown="1"></span>

6. **Modify CSS Attributes:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents16}
    ```javascript
    $("Selector").css("CSS-Attribute", "Value");
    ```
    > e.g. \\
    `$("p").css("color", "green");`

7. **Modify HTML Attributes:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents17}
    ```javascript
    $("Selector").attr("HTML-Attribute", "Value");
    ```
    > e.g. \\
    ``` $("a").attr("href", "https://en.wikipedia.org/wiki/Cats");```

    * **Deleting / Removing Attribute:**
        ```javascript
            $("Selector").removeAttr("HTML-Attribute");
        ```

8. **Add Class Name Attribute:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents18}
    ```javascript
    $("Selector").addClass("className");
    ```
    > e.g. \\
    ``` $("h2").addClass("header2");```

9. **Creating DOM Elements:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents19}
    ```javascript
    $("<TagName>");
    ```
    > e.g. \\
    `var paragraph = $("<p>");`
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

    * <button>Click for Video</button>{: .video_buttons value="show"
    src="https://www.youtubeeducation.com/embed/mboB93fon3w" onclick="iframePopInject(event);"}
        <div markdown="1"> </div>

10. **Looping through Elements / for loops:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents110} \\
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

    <button>Click for Video</button>{: #video_buttons110 .video_buttons value="show"
    src="https://www.youtubeeducation.com/embed/-IEnvFv5eBQ" onclick="iframePopInject(event);"}
        <span markdown="1"></span>

11. **Chaining:**{: style="color: SteelBlue"}{: .bodyContents1 #bodyContents111}
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

    This is possible because most *jQuery*  functions return *jQuery Collections*.

    <button>Click for Video</button>{: #video_buttons111 .video_buttons value="show"
    src="https://www.youtubeeducation.com/embed/9w9FB6J6euA" onclick="iframePopInject(event);"}
        <span markdown="1"></span>


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

    <button>Click for Video</button>{: #video_buttons21 .video_buttons value="show"
    src="https://www.youtubeeducation.com/embed/mV1pOhx4heI" onclick="iframePopInject(event);"}
        <span markdown="1"></span>

2. **Triggering events (Manually):**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents22} \\
    `$("#save-button").trigger("click");`

3. **Using Event Properties:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents23} \\
    <button>Click for Video</button>{: .video_buttons value="show"
    src="https://www.youtubeeducation.com/embed/UG-gTkmxDxE" onclick="iframePopInject(event);"}
        <span markdown="1"></span>

4. **Event Properties:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents24} \\
    `event.Property;`  
    <button>Click for list of Event Properties</button>{: .showText value="show" onclick="showTextPopHide(event);"}
    * **target:** The DOM element that initiated the event.
    * **relatedTarget:** The other DOM element involved in the event, if any.
    * **pageX:** The mouse position relative to the left edge of the document.
    * **pageY:** The mouse position relative to the top edge of the document.
    * **which:** For key or mouse events, this property indicates the specific key or button that was pressed.
    * **metaKey:** Indicates whether the META key was pressed when the event fired.
    * **preventDefault:** If this method is called, the default action of the event will not be triggered.
    * **event.type:** Describes the nature of the event.  
    {: hidden=""}


5. **Types of Events:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents25}  
    <button>Click for list of Events</button>{: input="<ul><li>target</li></ul>"
    .showText value="show" onclick="showTextPopHide(event);"}
    * **mouse events**: 'click' , 'mousedown'/'mouseup', 'mousemove', 'mouseenter'/'mouseleave'
    * **keyboard events**: 'keydown', 'keypress', 'keyup'
    * **touch events**: 'touchdown', 'touchup', 'touchstart', ‘touchcancel’
    * **drag events**: 'dragstart', 'dragend' (Many developers use jQueryUI for drag functionality, as it can be tricky to use the drag events directly.)
    * **form events**: 'submit', 'change', 'focus'/'blur'
    * **window events**: 'scroll', 'reload', 'hashchange'
    {: hidden=""}

    > source \\
    > [Khan Academy](https://www.khanacademy.org/computing/computer-programming/html-js-jquery/dom-events-with-jquery/a/dom-events-and-properties)[^3]

6. **Checking DOM Readiness:**{: style="color: SteelBlue"}{: .bodyContents2 #bodyContents26}
    ```javascript
    $(document).ready(function() {
        someFunction();
    });
    ```

    <button>Click for Video</button>{: .video_buttons value="show"
    src="https://www.youtubeeducation.com/embed/x7TMXXxsO3E" onclick="iframePopInject(event);"}
        <span markdown="1"></span>

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
