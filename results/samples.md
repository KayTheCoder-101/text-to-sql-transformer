# Qualitative samples (dev)

Predictions: `dev_beam_a0.6_avg5.jsonl`. "Correct" means the logical form matches the gold query.

## Correct

### dev #0
**Question:** What position does the player who played for butler cc (ks) play?  
**Gold SQL:** `SELECT Position FROM table WHERE School/Club Team = 'Butler CC (KS)'`  
**Our SQL:** `SELECT Position FROM table WHERE School/Club Team = 'butler cc (ks)'`  

### dev #1
**Question:** How many schools did player number 3 play at?  
**Gold SQL:** `SELECT COUNT(School/Club Team) FROM table WHERE No. = '3'`  
**Our SQL:** `SELECT COUNT(School/Club Team) FROM table WHERE No. = '3'`  

### dev #2
**Question:** What school did player number 21 play for?  
**Gold SQL:** `SELECT School/Club Team FROM table WHERE No. = '21'`  
**Our SQL:** `SELECT School/Club Team FROM table WHERE No. = '21'`  

### dev #3
**Question:** Who is the player that wears number 42?  
**Gold SQL:** `SELECT Player FROM table WHERE No. = '42'`  
**Our SQL:** `SELECT Player FROM table WHERE No. = '42'`  

### dev #5
**Question:** Who are all of the players on the Westchester High School club team?  
**Gold SQL:** `SELECT Player FROM table WHERE School/Club Team = 'Westchester High School'`  
**Our SQL:** `SELECT Player FROM table WHERE School/Club Team = 'westchester high school'`  

## Wrong

### dev #4
**Question:** What player played guard for toronto in 1996-97?  
**Gold SQL:** `SELECT Player FROM table WHERE Position = 'Guard' AND Years in Toronto = '1996-97'`  
**Our SQL:** `SELECT Player FROM table WHERE Nationality = 'guard' AND Years in Toronto = '1996-97'`  
**Failure:** wrong condition column  

### dev #7
**Question:** What are the total amount of numbers on the Toronto team in 2005-06?  
**Gold SQL:** `SELECT COUNT(No.) FROM table WHERE Years in Toronto = '2005-06'`  
**Our SQL:** `SELECT COUNT(Years in Toronto) FROM table WHERE School/Club Team = '2005-06'`  
**Failure:** wrong column (SELECT), wrong condition column  

### dev #20
**Question:** What is the episode number that has production code 8abx15?  
**Gold SQL:** `SELECT MIN(No. in series) FROM table WHERE Production code = '8ABX15'`  
**Our SQL:** `SELECT No. in series FROM table WHERE Original air date = '8abx15'`  
**Failure:** wrong aggregation, wrong condition column  

### dev #22
**Question:** What is the amount of trees, that require replacement when prevailing types, % is pine — 29.37 poplar — 26.12 acer negundo — 13.2?  
**Gold SQL:** `SELECT Amount of trees, that require replacement FROM table WHERE Prevailing types, % = 'Pine — 29.37 Poplar — 26.12 Acer negundo — 13.2'`  
**Our SQL:** `SELECT Amount of trees, that require replacement FROM table WHERE Amount of old trees = 'prevailing' AND Prevailing types, % = '— 26.12 acer negundo — 13.2'`  
**Failure:** extra condition  

### dev #62
**Question:** Provide me with the name of all the village (German) that are part of the village (Slovenian) with sele srednji kot.   
**Gold SQL:** `SELECT Village (German) FROM table WHERE Village (Slovenian) = 'Sele Srednji Kot'`  
**Our SQL:** `SELECT Village (German) FROM table WHERE Village (Slovenian) = 'sele srednji kot.'`  
**Failure:** wrong value  

