---
layout: NotesPage
title: AJAX
permalink: /work_files/web_dev/Ajax_notes
prevLink: /work_files/web_dev.html
---

# Notes

1. **Asynchronous:**{: style="color: SteelBlue  "}{: .bodyContents1 #bodyContents11} \\
    The AJAX function returns before the success callback is called/returned.
2. **Ajax Function Returns:**{: style="color: SteelBlue  "}{: .bodyContents1 #bodyContents12} \\
    The `$.ajax()` function returns the XMLHttpRequest object that it creates.
3. **Call Back Functions:**{: style="color: SteelBlue  "}{: .bodyContents1 #bodyContents13} \\
    * **JSON Response:**
    ```javascript
        function successCallback(responseObj){
            // Do something like read the response and show data
            alert(JSON.stringify(responseObj)); // Only applicable to JSON response
        }
    ```

***


1. **AJAX Example / Boiler-Plate:**{: style="color: SteelBlue  "}{: .bodyContents2 #bodyContents21} \\
    ```javascript
    // Create the "callback" functions that will be invoked when...

    // ... the AJAX request is successful
    var updatePage = function( resp ) {
      $( '#target').html( resp.people[0].name );
    };

    // ... the AJAX request fails
    var printError = function( req, status, err ) {
      console.log( 'something went wrong', status, err );
    };

    // Create an object to describe the AJAX request
    var ajaxOptions = {
      url: '/data/people.json',
      dataType: 'json',
      success: updatePage,
      error: printError
    };

    // Initiate the request!
    $.ajax(ajaxOptions);
    ```