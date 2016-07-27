Each filename is the uuid of the user

If the userData file has 5 columns, then it was by top topic:
  The first column is the time (in milliseconds since Jan 1, 1970)
    when the user started rating the document
  The second column is the time (in milliseconds since Jan 1, 1970)
    when the user finished rating the document
  The third column is the topic of the document
  The fourth column is the document number (from the first column in amazon.txt)
  The fifth column is the star rating the user guessed for the review

If the userData file has 4 columns, then it was by JS Divergence:
  The first column is the time (in milliseconds since Jan 1, 1970)
    when the user started rating the document
  The second column is the time (in milliseconds since Jan 1, 1970)
    when the user finished rating the document
  The third column is the document number (from the first column in amazon.txt)
  The fourth column is the star rating the user guessed for the review
