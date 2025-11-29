
## QUICK START:

```
#
# To initialize
#
brew install just                      # installs just
just init                              # creates virtualenv, installs packages
python -m textblob.download_corpora    # installs words for misspelling checks
alias tags='just run'                  # bash alias for convenience
```

```
#
# To analyze the most-recent works file from the internet
#
tags --download
```

```
#
# To analyze the 'works.txt' file stored locally
#
tags
```

```
#
# To analyze a local file that has some other name
#
tags --filename works-foo.txt
```
