# Error analysis - detector `llm` on HaluEval test (threshold 0.500)


## False positives: correct answers flagged as hallucinated (3 shown)

- **Q:** Which two countries face each other in a cricket match every year?
  - answer: `England and Australia`  | score 0.500
  - claim(s): ['England faces Australia in a cricket match every year.']
  - top evidence: ['The 2015 Ashes was a series of Test cricket matches played between England and Australia for The Ashes.The 2013–14 Ashes series, known for sponsorship reasons as the Commonwealth Bank Ashes Series, was a Test cricket series between England and Australia.']
  - gold paragraph: The 2015 Ashes was a series of Test cricket matches played between England and Australia for The Ashes.The 2013–14 Ashes series, known for sponsorship reasons as the Commonwealth Bank Ashes Series, was a Test cricket series between England and Australia.

- **Q:** What famous movie series did the star of BBC's One Child play a supporting role in?
  - answer: `Harry Potter`  | score 0.500
  - claim(s): ["The star of BBC's One Child played a supporting role in the Harry Potter movie series."]
  - top evidence: ['is a 2015 British independent family film written and directed by Tim Clague and Danny Stack and starring Bonnie Wright. She is best known for playing the role of Ginny Weasley in the "Harry Potter" film series, based on the "Harry Potter" novel series by British author J. K. Rowling.']
  - gold paragraph: The series follows a girl called Mei Ashley (Katie Leung) who discovers that she has a brother, Li Jun (Sebastian So) that she never knew about and that he is due to be sentenced to death for a murder he did not commit. She played Cho Chang, the first love interest for lead character Harry Potter in

- **Q:** Wonderland is from What Taylor Swift album that was released on October 27, 2014?
  - answer: `1989`  | score 0.500
  - claim(s): ["The song 'Wonderland' is from the album 1989."]
  - top evidence: ['Spot Marks the X is a 1986 Disney Channel Original Movie that starred Mike the Dog (from "Down and Out in Beverly Hills") as Capone/Astro, Barret Oliver (from "The Neverending Story", "Frankenweenie", "D.A.R.Y.L", and "Cocoon") as Ken, and Natalie Gregory (from "Alice in Wonderland (1985 version)"
  - gold paragraph: It was released as the third promotional single on February 17, 2015 from Swift's deluxe version of her fifth studio album, "1989".1989 is the fifth studio album by American singer-songwriter Taylor Swift released on October 27, 2014, through Big Machine Records.


## False negatives: hallucinations that were missed (1 shown)

- **Q:** Who wrote a letter to Orville Browning where he recognized the importance of one of the four slave states that never declared a secession in the Civil War?
  - answer: `In September 1861, a letter was written by a U.S President to Orville Browning recognizing the importance of Kentucky during Civil War.`  | score 0.000
  - claim(s): ['In September 1861, a U.S. President wrote a letter to Orville Browning.', 'The letter recognized the importance of Kentucky during the Civil War.']
  - top evidence: ['Four slave states never declared a secession: Delaware, Kentucky, Maryland, and Missouri.Kentucky was a border state of key importance in the American Civil War. President Abraham Lincoln recognized the importance of the Commonwealth when, in a September 1861 letter to Orville Browning, he wrote:'
  - gold paragraph: Four slave states never declared a secession: Delaware, Kentucky, Maryland, and Missouri.Kentucky was a border state of key importance in the American Civil War. President Abraham Lincoln recognized the importance of the Commonwealth when, in a September 1861 letter to Orville Browning, he wrote:
