"""ARPABET to IPA mapping for US - English phonemes"""

ARPABET_TO_IPA = {
    "AA": "ɑ",
    "AE": "æ",
    "AH0": "ə",
    "AH1": "ʌ",
    "AO": "ɔ",
    "AW": "aʊ",
    "AY": "aɪ",
    "B": "b",
    "CH": "tʃ",
    "D": "d",
    "DH": "ð",
    "EH": "ɛ",
    "ER": "ɚ",
    "EY": "eɪ",
    "F": "f",
    "G": "g",
    "HH": "h",
    "IH": "ɪ",
    "IY": "i",
    "JH": "dʒ",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "ŋ",
    "OW": "oʊ",
    "OY": "ɔɪ",
    "P": "p",
    "R": "r",
    "S": "s",
    "SH": "ʃ",
    "T": "t",
    "TH": "θ",
    "UH": "ʊ",
    "UW": "u",
    "V": "v",
    "W": "w",
    "Y": "j",
    "Z": "z",
    "ZH": "ʒ",
}


VOWEL_INITIALS = set("aeiouæɑɔɛɪʊʌəɚɝ")

"""Legacy Letters & Sounds curriculum aligned GPC inventory"""
GPC_INVENTORY: list[tuple[str, str, tuple[str, ...]]] = [
    ("G001", "s", ("s",)),
    ("G002", "a", ("æ",)),
    ("G003", "t", ("t",)),
    ("G004", "p", ("p",)),
    ("G005", "i", ("ɪ",)),
    ("G006", "n", ("n",)),
    ("G007", "m", ("m",)),
    ("G008", "d", ("d",)),
    ("G009", "g", ("g",)),
    ("G010", "o", ("ɑ",)),
    ("G011", "c", ("k",)),
    ("G012", "k", ("k",)),
    ("G013", "ck", ("k",)),
    ("G014", "e", ("ɛ",)),
    ("G015", "u", ("ʌ",)),
    ("G016", "r", ("r",)),
    ("G017", "h", ("h",)),
    ("G018", "b", ("b",)),
    ("G019", "f", ("f",)),
    ("G020", "ff", ("f",)),
    ("G021", "l", ("l",)),
    ("G022", "ll", ("l",)),
    ("G023", "ss", ("s",)),
    ("G024", "s", ("z",)),
    ("G025", "j", ("dʒ",)),
    ("G026", "v", ("v",)),
    ("G027", "w", ("w",)),
    ("G028", "x", ("k", "s")),
    ("G029", "y", ("j",)),
    ("G030", "z", ("z",)),
    ("G031", "zz", ("z",)),
    ("G032", "qu", ("k", "w")),
    ("G033", "ch", ("tʃ",)),
    ("G034", "sh", ("ʃ",)),
    ("G035", "th", ("θ",)),
    ("G036", "th", ("ð",)),
    ("G037", "ng", ("ŋ",)),
    ("G038", "ai", ("eɪ",)),
    ("G039", "ee", ("i",)),
    ("G040", "igh", ("aɪ",)),
    ("G041", "oa", ("oʊ",)),
    ("G042", "oo", ("u",)),
    ("G043", "oo", ("ʊ",)),
    ("G044", "ar", ("ɑ", "r")),
    ("G045", "or", ("ɔ", "r")),
    ("G046", "ur", ("ɚ",)),
    ("G047", "ow", ("aʊ",)),
    ("G048", "oi", ("ɔɪ",)),
    ("G049", "ear", ("ɪ", "r")),
    ("G050", "air", ("ɛ", "r")),
    ("G051", "ure", ("ʊ", "r")),
    ("G052", "er", ("ɚ",)),
    ("G053", "ay", ("eɪ",)),
    ("G054", "ou", ("aʊ",)),
    ("G055", "ie", ("aɪ",)),
    ("G056", "ea", ("i",)),
    ("G057", "oy", ("ɔɪ",)),
    ("G058", "ir", ("ɚ",)),
    ("G059", "ue", ("u",)),
    ("G060", "ue", ("j", "u")),
    ("G061", "aw", ("ɔ",)),
    ("G062", "wh", ("w",)),
    ("G063", "ph", ("f",)),
    ("G064", "ew", ("u",)),
    ("G065", "ew", ("j", "u")),
    ("G066", "oe", ("oʊ",)),
    ("G067", "au", ("ɔ",)),
    ("G068", "a-e", ("eɪ",)),
    ("G069", "e-e", ("i",)),
    ("G070", "i-e", ("aɪ",)),
    ("G071", "o-e", ("oʊ",)),
    ("G072", "u-e", ("u",)),
    ("G073", "u-e", ("j", "u")),
    ("G074", "i", ("aɪ",)),
    ("G075", "o", ("oʊ",)),
    ("G076", "c", ("s",)),
    ("G077", "g", ("dʒ",)),
    ("G078", "u", ("ʊ",)),
    ("G079", "ow", ("oʊ",)),
    ("G080", "ie", ("i",)),
    ("G081", "ea", ("ɛ",)),
    ("G082", "er", ("ɚ",)),
    ("G083", "y", ("aɪ",)),
    ("G084", "y", ("i",)),
    ("G085", "ch", ("k",)),
    ("G086", "ch", ("ʃ",)),
    ("G087", "ou", ("oʊ",)),
    ("G088", "ou", ("ʊ",)),
    ("G089", "ou", ("u",)),
    ("G090", "a", ("ɑ",)),
]

"""Ultimate Phonics reading program aligned GPC inventory (Lessons 1-209).
CMUdict-based phoneme transcriptions are normalized to match the phonics
program's expectations."""
UPL_GPC_INVENTORY: list[tuple[str, str, tuple[str, ...]]] = [
    # === Short Vowels (Lessons 2-12) ===
    ("UP001", "a", ("æ",)),  # cat, sat, hat
    ("UP002", "i", ("ɪ",)),  # sit, pin, big
    ("UP003", "u", ("ʌ",)),  # cup, but, run
    ("UP004", "e", ("ɛ",)),  # bed, get, pen
    ("UP005", "o", ("ɑ",)),  # hot, pot, dog
    # === Single Consonants (Lesson 1) ===
    ("UP006", "b", ("b",)),  # bat, bin, bed
    ("UP007", "c", ("k",)),  # cat, cup, cap
    ("UP008", "d", ("d",)),  # dad, dip, dog
    ("UP009", "f", ("f",)),  # fit, fun, fat
    ("UP010", "g", ("g",)),  # got, gap, gas
    ("UP011", "h", ("h",)),  # hat, him, hid
    ("UP012", "j", ("dʒ",)),  # jam, jet, jab
    ("UP013", "k", ("k",)),  # kit, keg, kid
    ("UP014", "l", ("l",)),  # let, lip, lit
    ("UP015", "m", ("m",)),  # mat, men, map
    ("UP016", "n", ("n",)),  # nap, net, nod
    ("UP017", "p", ("p",)),  # pat, pin, pot
    ("UP018", "r", ("r",)),  # run, red, rat
    ("UP019", "s", ("s",)),  # sat, sip, sun
    ("UP020", "t", ("t",)),  # tap, tin, top
    ("UP021", "v", ("v",)),  # van, vet, vat
    ("UP022", "w", ("w",)),  # wet, win, wag
    ("UP023", "x", ("k", "s")),  # fox, box, six — 2 phonemes
    ("UP024", "y", ("j",)),  # yes, yet, yam
    ("UP025", "z", ("z",)),  # zip, zoo, zap
    # === Consonant Digraphs & Trigraphs (Lessons 52-65) ===
    ("UP026", "sh", ("ʃ",)),  # ship, shop, fish, rush
    ("UP027", "ch", ("tʃ",)),  # chip, much, lunch, branch
    ("UP028", "tch", ("tʃ",)),  # catch, match, itch, sketch
    ("UP029", "th", ("θ",)),  # thin, bath, math, cloth
    ("UP030", "th", ("ð",)),  # that, them, this, smooth
    ("UP031", "wh", ("w",)),  # when, whip, whiz, which
    ("UP032", "ck", ("k",)),  # back, duck, kick, rock
    # === Double Consonants (Lessons 66-69) ===
    ("UP033", "ff", ("f",)),  # off, puff, cliff, staff
    ("UP034", "ss", ("s",)),  # miss, fuss, dress, boss
    ("UP035", "ll", ("l",)),  # bell, fill, dull, skill
    ("UP036", "bb", ("b",)),  # ebb, rabbit
    ("UP037", "gg", ("g",)),  # egg, bigger
    ("UP038", "dd", ("d",)),  # add, odd
    ("UP039", "nn", ("n",)),  # inn, penny, Ann
    ("UP040", "tt", ("t",)),  # butt, mutt, kitten
    ("UP041", "zz", ("z",)),  # buzz, fuzz, jazz, fizz
    # === qu, ng, nk (Lessons 70-80) ===
    ("UP042", "qu", ("k", "w")),  # quit, quiz, queen — 2 phonemes
    ("UP043", "ng", ("ŋ",)),  # ring, song, sing, hung
    ("UP044", "nk", ("ŋ", "k")),  # ink, think, bank — 2 phonemes
    # === Vowel Digraphs (Lessons 82-106) ===
    ("UP045", "ee", ("i",)),  # see, bee, tree, sleep, green
    ("UP046", "ea", ("i",)),  # eat, sea, read, teach, dream
    ("UP047", "oo", ("u",)),  # moon, food, boot, room, zoo
    ("UP048", "oo", ("ʊ",)),  # book, cook, look, good, foot
    ("UP049", "ai", ("eɪ",)),  # rain, wait, train, paint, sail
    ("UP050", "air", ("ɛ", "r")),  # air, hair, fair, chair, stair
    ("UP051", "ay", ("eɪ",)),  # day, play, say, stay, gray
    ("UP052", "oa", ("oʊ",)),  # boat, coat, road, toast, float
    ("UP053", "oar", ("ɔ", "r")),  # oar, soar, roar, board, hoard
    ("UP054", "ou", ("aʊ",)),  # out, loud, house, cloud, pound
    ("UP055", "ow", ("aʊ",)),  # cow, now, town, brown, crowd
    ("UP056", "aw", ("ɔ",)),  # saw, draw, lawn, crawl, hawk
    ("UP057", "au", ("ɔ",)),  # Paul, haul, fault, launch, sauce
    ("UP058", "oy", ("ɔɪ",)),  # boy, toy, joy, enjoy
    ("UP059", "oi", ("ɔɪ",)),  # oil, coin, join, point, soil
    # === Ending Patterns (Lessons 108-124) ===
    ("UP060", "se", ("z",)),  # house, goose, geese, breeze
    ("UP061", "se", ("s",)),  # moose, rinse, sense, tense
    ("UP062", "ce", ("s",)),  # dance, fence, prince, peace
    ("UP063", "ge", ("dʒ",)),  # large, change, range, lounge
    ("UP064", "ve", ("v",)),  # have, give, live, love, glove
    ("UP065", "s", ("z",)),  # is, has, as, please, raise
    # === Suffixes (Lessons 116-122) ===
    ("UP066", "ed", ("ɪ", "d")),  # [CMU] needed — lesson /ɛd/, CMU /ɪd/
    ("UP067", "ed", ("d",)),  # filled, played, cleaned
    ("UP068", "ed", ("t",)),  # asked, jumped, missed
    ("UP069", "ful", ("f", "ʊ", "l")),  # [CMU] helpful — lesson /fəl/, CMU /fʊl/
    # === Multisyllable Patterns (Lessons 125-145) ===
    ("UP070", "a", ("ə",)),  # away, about, banana — vowel reduction
    ("UP071", "al", ("ə", "l")),  # medal, animal, capital
    ("UP072", "er", ("ɚ",)),  # after, winter, dinner, better
    ("UP073", "le", ("ə", "l")),  # table, little, apple, puzzle
    ("UP074", "tle", ("ə", "l")),  # castle, whistle, bristle — silent t
    ("UP075", "y", ("i",)),  # happy, funny, candy, family
    ("UP076", "ly", ("l", "i")),  # badly, softly, quickly
    # === Soft C and G (Lessons 148-155) ===
    ("UP077", "c", ("s",)),  # cent, city, cell, pencil
    ("UP078", "g", ("dʒ",)),  # gem, giant, gentle, magic
    ("UP079", "age", ("ɪ", "dʒ")),  # image, manage, village, package
    ("UP080", "dge", ("dʒ",)),  # bridge, badge, fudge, judge
    # === Split Digraphs (Lessons 156-171) ===
    ("UP081", "a-e", ("eɪ",)),  # make, cake, name, game, brave
    ("UP082", "i-e", ("aɪ",)),  # time, like, five, ride, smile
    ("UP083", "o-e", ("oʊ",)),  # home, bone, joke, hope, note
    ("UP084", "u-e", ("u",)),  # rule, rude, June, tube, flute
    ("UP085", "u-e", ("j", "u")),  # use, fuse, cube, cute, mule
    ("UP086", "e-e", ("i",)),  # eve, here, these, theme, Pete
    ("UP087", "are", ("ɛ", "r")),  # bare, care, dare, share, square
    ("UP088", "ore", ("ɔ", "r")),  # more, bore, core, score, store
    # === Open Syllable Vowels (Lessons 172-183) ===
    ("UP089", "a", ("eɪ",)),  # able, baby, table, basic, April
    ("UP090", "e", ("i",)),  # be, she, he, begin, even, zebra
    ("UP091", "i", ("aɪ",)),  # item, ivy, tiger, spider, Friday
    ("UP092", "o", ("oʊ",)),  # open, over, bonus, pony, tomato
    ("UP093", "u", ("j", "u")),  # unit, music, human, pupil, menu
    ("UP094", "u", ("u",)),  # flu, tuna, duty, truly, super
    ("UP095", "i", ("i",)),  # piano, radio, patio, medium
    # === More Long Vowels (Lessons 184-209) ===
    ("UP096", "ie", ("i",)),  # chief, field, thief, believe
    ("UP097", "ie", ("aɪ",)),  # die, lie, pie, tie, tries
    ("UP098", "ei", ("i",)),  # seize, ceiling, receive, deceive
    ("UP099", "ey", ("i",)),  # key, donkey, chimney, hockey
    ("UP100", "y", ("aɪ",)),  # cry, fly, dry, sky, try, July
    ("UP101", "igh", ("aɪ",)),  # high, night, light, right, bright
    ("UP102", "oe", ("oʊ",)),  # toe, foe, doe, hoe, goes
    ("UP103", "ow", ("oʊ",)),  # own, low, grow, snow, throw
    ("UP104", "ou", ("oʊ",)),  # pour, four, court, soul, shoulder
    ("UP105", "ue", ("u",)),  # true, blue, clue, glue, avenue
    ("UP106", "ue", ("j", "u")),  # cue, hue, value, rescue
    ("UP107", "ew", ("u",)),  # new, drew, grew, crew, threw
    ("UP108", "ew", ("j", "u")),  # few, mew, hew, pew, skew
    ("UP109", "ui", ("u",)),  # fruit, juice, suit, cruise
    ("UP110", "ou", ("u",)),  # soup, group, youth, route
    ("UP111", "eigh", ("eɪ",)),  # eight, weigh, weight, freight
    ("UP112", "ei", ("eɪ",)),  # vein, rein, reindeer
    ("UP113", "ey", ("eɪ",)),  # hey, prey, grey, obey, convey
    # === R-Controlled Syllables (Lessons 210-225) ===
    ("UP114", "ar", ("ɑ", "r")),  # car, park, farm, art, garden
    ("UP115", "ir", ("ɚ",)),  # bird, girl, stir, dirt, circle
    ("UP116", "or", ("ɔ", "r")),  # fork, corn, horse, doctor
    ("UP117", "ur", ("ɚ",)),  # fur, burn, nurse, church, purple
    ("UP118", "ear", ("ɚ",)),  # earth, earn, learn, search
    ("UP119", "wor", ("w", "ɚ")),  # word, work, worm, world, worse
    ("UP120", "war", ("w", "ɔ", "r")),  # war, warm, warn, award
    ("UP121", "quar", ("k", "w", "ɔ", "r")),  # quart, quarter, quarrel
    # === More Short Vowels (Lessons 226-239) ===
    ("UP122", "all", ("ɔ", "l")),  # ball, call, tall, wall, small
    ("UP123", "al", ("ɔ", "l")),  # also, always, almost, salt
    ("UP124", "alk", ("ɔ", "k")),  # talk, walk, chalk, stalk
    ("UP125", "a", ("ɑ",)),  # want, wash, watch, water — after w/qu
    ("UP126", "ea", ("ɛ",)),  # bread, head, dead, weather
    ("UP127", "ear", ("ɛ", "r")),  # bear, pear, tear, wear, swear
    ("UP128", "y", ("ɪ",)),  # gym, myth, system, symbol
    ("UP129", "ous", ("ə", "s")),  # famous, nervous, curious
    ("UP130", "ou", ("ʌ",)),  # touch, double, trouble, country
    ("UP131", "our", ("ɚ",)),  # journey, courage, nourish
    ("UP132", "o", ("ʌ",)),  # mother, brother, come, some, love
    # === Silent Letters & Advanced (Lessons 240-262) ===
    ("UP133", "oo", ("ʌ",)),  # blood, flood — rare irregular
    ("UP134", "gh", ("f",)),  # enough, cough, rough, tough
    ("UP135", "ough", ("ɔ",)),  # thought, bought, fought, brought
    ("UP136", "sc", ("s",)),  # scene, scent, science, scissors
    ("UP137", "wr", ("r",)),  # wrap, write, wrong, wrist
    ("UP138", "kn", ("n",)),  # knee, knit, know, knock, knife
    ("UP139", "gn", ("n",)),  # gnat, gnaw, gnu, gnome, sign
    ("UP140", "ign", ("aɪ", "n")),  # sign, design, assign, resign
    ("UP141", "mb", ("m",)),  # bomb, lamb, limb, climb, thumb
    ("UP142", "mn", ("m",)),  # autumn, column, hymn, solemn
    ("UP143", "bt", ("t",)),  # debt, doubt
    ("UP144", "ps", ("s",)),  # psalm, psychology
    ("UP145", "lk", ("k",)),  # talk, walk, folk — silent l
    ("UP146", "lf", ("f",)),  # half, calf — silent l
    ("UP147", "lm", ("m",)),  # calm, palm, balm — silent l
    ("UP148", "ch", ("k",)),  # school, echo, ache, stomach
    ("UP149", "ph", ("f",)),  # phone, photo, graph, elephant
    ("UP150", "ch", ("ʃ",)),  # chef, machine, brochure
    ("UP151", "tion", ("ʃ", "ə", "n")),  # action, nation, station
    ("UP152", "sion", ("ʃ", "ə", "n")),  # mansion, mission, passion
    ("UP153", "sion", ("ʒ", "ə", "n")),  # vision, decision, television
    ("UP154", "ci", ("ʃ",)),  # special, social, official
    ("UP155", "ti", ("ʃ",)),  # patient, partial, caution
    ("UP156", "ture", ("tʃ", "ɚ")),  # picture, nature, future
    ("UP158", "i", ("ə",)),  # verify, animal, pencil
    ("UP159", "e", ("ə",)),  # taken, open, item
    ("UP160", "o", ("ə",)),  # official, lemon, common
    ("UP161", "u", ("ə",)),  # supply, circus, lettuce
    ("UP162", "o", ("ɔ",)),  # [CMU] dog, log — cot-caught unmerged
    ("UP163", "ze", ("z",)),  # breeze, freeze, sneeze, squeeze
    ("UP164", "the", ("ð",)),  # breathe, bathe, soothe
    ("UP165", "aw", ("ɑ",)),  # [CMU] law, raw — cot-caught merged
    ("UP166", "pp", ("p",)),  # hopped, stopped, slipped, trapped
    ("UP167", "mm", ("m",)),  # humming, swimming, hammer
    ("UP168", "es", ("ɪ", "z")),  # [CMU] misses, dishes — CMU /ɪz/ not /ɛz/
    ("UP169", "'s", ("z",)),  # he's, she's, who's — after voiced
    ("UP170", "'s", ("s",)),  # it's, let's, that's, what's — after voiceless
    ("UP171", "'d", ("d",)),  # I'd, he'd, we'd, you'd
    ("UP172", "'m", ("m",)),  # I'm
    ("UP173", "'ll", ("l",)),  # I'll, he'll, we'll, they'll
    ("UP174", "'ve", ("v",)),  # I've, we've, you've
    ("UP175", "'re", ("r",)),  # you're, they're
    ("UP176", "'re", ("ɚ",)),  # [CMU] we're — r-colored syllable
    ("UP177", "n't", ("n", "t")),  # didn't, aren't, weren't
    ("UP178", "n't", ("ə", "n", "t")),  # [CMU] hasn't, wouldn't — schwa inserted
    ("UP179", "n't", ("ə", "n")),  # [CMU] hadn't, doesn't — schwa + silent t
    ("UP180", "e", ("ɪ",)),  # [CMU] given, jacket — weak vowel merger
    ("UP181", "ng", ("ŋ", "g")),  # [CMU] finger, anger — lesson /ŋ/, CMU /ŋg/
    ("UP182", "rr", ("r",)),  # berry, carry, sorry — double consonant class
    ("UP183", "a", ("ɛ",)),  # [CMU] carry, marry, parent — Mary-marry-merry merger
    ("UP184", "x", ("k",)),  # excel, except, excite — before soft c
    ("UP185", "ire", ("aɪ", "ɚ")),  # tire, entire, desire, require
    ("UP186", "ure", ("j", "ʊ", "r")),  # pure, cure — yod + short u + r
    ("UP187", "ure", ("j", "ɚ")),  # figure — reduced yod + r-colored
    ("UP188", "dg", ("dʒ",)),  # budget, gadget, midget, badger
    ("UP189", "ar", ("ɚ",)),  # Lesson 213 unstressed -ar: beggar, dollar, hazard
    ("UP190", "or", ("ɚ",)),  # Lesson 219 unstressed -or: actor, doctor, favor
    ("UP191", "our", ("ɔ", "r")),  # Lesson 197: four, court, course, source
    ("UP192", "al", ("ɔ", "l")),  # Lesson 227: always, waltz, salt
    ("UP193", "a", ("ɔ",)),  # Lesson 229 a=/ŏ/ after w: want, watch, water
    ("UP194", "wor", ("w", "ɚ")),  # Lesson 223: worry, word, world
    ("UP195", "y-e", ("aɪ",)),  # Lesson 190 magic-e on y: type, style
    ("UP196", "ia", ("aɪ",)),  # Lesson 182 split vowel pairs: dial, trial
    ("UP197", "u", ("ʊ",)),  # Lesson 231 u = short oo: put, push, bull, full
    ("UP198", "ous", ("ə", "s")),  # Lesson 236 ous = /us/: fabulous, curious
    ("UP199", "wr", ("r",)),  # Lesson 242 wr = /r/: writhe, write, wreck
    ("UP200", "augh", ("ɔ",)),  # Lesson 246 gh silent: caught, taught, daughter
    ("UP201", "augh", ("æ", "f")),  # Lesson 246 gh = /f/: laugh, laughter
    ("UP202", "aigh", ("eɪ",)),  # Lesson 246 gh silent: straight, straighten
    ("UP203", "ough", ("aʊ",)),  # Lesson 246 gh silent: bough
    ("UP204", "ough", ("ɔ", "f")),  # Lesson 246 gh = /f/: cough, trough
    ("UP205", "gh", ("g",)),  # Lesson 249 silent h: ghost, ghastly, ghetto
    ("UP206", "gu", ("g",)),  # Lesson 250 silent u: guess, guilt, guitar, guide
    ("UP207", "gue", ("g",)),  # Lesson 250 silent ue: league, vague, rogue
    ("UP208", "ui", ("ɪ",)),  # Lesson 251 silent u: build, biscuit, circuit
    ("UP209", "ie", ("ɛ",)),  # Lesson 251 silent i: friend, friendly
    ("UP210", "sten", ("s", "ə", "n")),  # Lesson 249 silent t: soften, listen, fasten
    ("UP211", "tion", ("ʃ", "ə", "n")),  # Lesson 255 tion = /shun/: direction
    ("UP212", "sion", ("ʃ", "ə", "n")),  # Lesson 257 sion = /shun/: mission
    ("UP213", "ssion", ("ʃ", "ə", "n")),  # Lesson 257 ssion = /shun/: mission, passion
    ("UP214", "cian", ("ʃ", "ə", "n")),  # Lesson 259 cian = /shun/: physician, musician
    ("UP215", "rh", ("r",)),  # Lesson 249 silent h: rhyme, rhythm
    ("UP216", "ture", ("ʃ", "ɚ")),  # [CMU] fracture — CMU drops /t/ from /tʃɚ/
    ("UP217", "tion", ("tʃ", "ə", "n")),  # [CMU] intention — CMU /tʃ/ not /ʃ/
    ("UP218", "sion", ("tʃ", "ə", "n")),  # [CMU] expansion — CMU /tʃ/ not /ʃ/
]

UPL_GPC_PAIRS: frozenset[tuple[str, tuple[str, ...]]] = frozenset(
    (grapheme, phonemes) for _, grapheme, phonemes in UPL_GPC_INVENTORY
)
