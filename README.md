# lshensemble
LSH index for approximate set containment search. Forked from the original repository to make use of the index for our specific use-case.
Only the main script was edited, which is why this is the only file in this repository.

The script requires two inputs:
* The first one is the input directory containing two types of json files (schema specified inside Main.go (the Domainstruct)), recognized by their name
  * If the name contains "index" the contained domains will be used to build the index
  * If the name contains "query" the contained domains will queries against the index
* The second argument is the output directory
