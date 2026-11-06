#!/usr/bin/env python3
"""
Expand the philosophy corpus with LONG passages chunked from
public-domain philosophy books (Project Gutenberg via the Gutendex API).

Unlike expand_corpus.py (short quotes from Wikiquote), this script pulls
entire books — Plato, the Analects, Kant, the Bhagavad Gita, … — and
splits them into 120–250 word passages, so every corpus entry is a
substantial philosophical text.

Run ON YOUR OWN COMPUTER (needs internet):

    python3 expand_corpus_books.py            # harvest the curated list
    python3 expand_corpus_books.py --stats    # show DB stats only

Per-country target: 1000 passages where authentic material exists.
Small nations genuinely lack 1000 digitized philosophy texts — the
script reports the real numbers instead of padding with fake data
(that gap is exactly the "why 120+ nations" limitation of the corpus).

All books are public domain (Project Gutenberg license headers are
stripped during chunking).
"""

import argparse
import json
import os
import re
import sqlite3
import time
import urllib.parse
import urllib.request

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, "philosophy_corpus.db")

USER_AGENT = (
    "PhilosophySemanticExplorer/1.0 "
    "(independent student research project)"
)

GUTENDEX = "https://gutendex.com/books"

PASSAGE_MIN_WORDS = 120
PASSAGE_MAX_WORDS = 250
MAX_PASSAGES_PER_BOOK = 1500
TARGET_PER_COUNTRY = 8000
REQUEST_PAUSE = 0.3

# (gutendex search query, author, work, country, region, tradition, era)
BOOKS = [
    # ---------------- China ----------------
    ("Analects Confucius Legge", "Confucius (tr. James Legge)",
     "The Analects", "China", "East Asia", "Confucianism", "5th c. BCE"),
    ("Tao Teh King Legge", "Laozi (tr. James Legge)", "Tao Te Ching",
     "China", "East Asia", "Daoism", "4th c. BCE"),
    ("Works of Mencius Legge", "Mencius (tr. James Legge)", "Mencius",
     "China", "East Asia", "Confucianism", "4th c. BCE"),
    ("Musings Chinese Mystic Chuang", "Zhuangzi (tr. Lionel Giles)",
     "Musings of a Chinese Mystic", "China", "East Asia", "Daoism",
     "4th c. BCE"),
    ("Art of War Sunzi Giles", "Sun Tzu (tr. Lionel Giles)",
     "The Art of War", "China", "East Asia", "Strategic philosophy",
     "5th c. BCE"),
    ("Ethical Political Works Motse", "Mozi (tr. Y. P. Mei)",
     "The Ethical and Political Works of Motse", "China", "East Asia",
     "Mohism", "5th c. BCE"),

    # ---------------- Japan ----------------
    ("Bushido Soul of Japan Nitobe", "Inazō Nitobe",
     "Bushido: The Soul of Japan", "Japan", "East Asia", "Bushido",
     "1900"),
    ("Japan Attempt at Interpretation Hearn", "Lafcadio Hearn",
     "Japan: An Attempt at Interpretation", "Japan", "East Asia",
     "Japanese culture studies", "1904"),

    # ---------------- India ----------------
    ("Bhagavad-Gita Arnold", "Vyasa (tr. Edwin Arnold)",
     "The Bhagavad Gita", "India", "South Asia", "Hindu philosophy",
     "c. 200 BCE"),
    ("Dhammapada Muller", "The Buddha (tr. Max Müller)", "Dhammapada",
     "India", "South Asia", "Buddhism", "3rd c. BCE"),
    ("Upanishads Sacred Books East", "Unknown (tr. Max Müller)",
     "The Upanishads", "India", "South Asia", "Vedanta", "c. 700 BCE"),
    ("Gitanjali Tagore", "Rabindranath Tagore", "Gitanjali", "India",
     "South Asia", "Bengali humanism", "1910"),
    ("Sadhana Tagore", "Rabindranath Tagore", "Sādhanā", "India",
     "South Asia", "Bengali humanism", "1913"),
    ("Raja Yoga Vivekananda", "Swami Vivekananda", "Raja Yoga", "India",
     "South Asia", "Vedanta", "1896"),
    ("Vedanta Philosophy lectures Vivekananda", "Swami Vivekananda",
     "Vedanta Philosophy: Five Lectures", "India", "South Asia",
     "Vedanta", "1893"),

    # ---------------- Greece ----------------
    ("Republic Plato Jowett", "Plato (tr. Benjamin Jowett)",
     "The Republic", "Greece", "Europe", "Ancient Greek philosophy",
     "c. 375 BCE"),
    ("Apology Crito Phaedo Plato", "Plato (tr. Benjamin Jowett)",
     "Apology, Crito, and Phaedo", "Greece", "Europe",
     "Ancient Greek philosophy", "4th c. BCE"),
    ("Symposium Plato", "Plato (tr. Benjamin Jowett)", "Symposium",
     "Greece", "Europe", "Ancient Greek philosophy", "c. 385 BCE"),
    ("Nicomachean Ethics Aristotle", "Aristotle",
     "Nicomachean Ethics", "Greece", "Europe",
     "Ancient Greek philosophy", "4th c. BCE"),
    ("Politics Aristotle", "Aristotle", "Politics", "Greece", "Europe",
     "Ancient Greek philosophy", "4th c. BCE"),
    ("Memorabilia Xenophon", "Xenophon", "Memorabilia", "Greece",
     "Europe", "Ancient Greek philosophy", "4th c. BCE"),
    ("Golden Sayings Epictetus", "Epictetus",
     "The Golden Sayings of Epictetus", "Greece", "Europe", "Stoicism",
     "c. 125 CE"),
    ("Meditations Marcus Aurelius", "Marcus Aurelius", "Meditations",
     "Italy", "Europe", "Stoicism", "2nd c. CE"),

    # ---------------- Rome / Italy ----------------
    ("Nature of Things Lucretius", "Lucretius",
     "Of the Nature of Things", "Italy", "Europe", "Epicureanism",
     "1st c. BCE"),
    ("Peace of Mind Seneca", "Seneca", "Of Peace of Mind", "Italy",
     "Europe", "Stoicism", "1st c. CE"),
    ("Friendship Cicero", "Cicero", "On Friendship", "Italy", "Europe",
     "Roman philosophy", "1st c. BCE"),
    ("Old Age Cicero", "Cicero", "On Old Age", "Italy", "Europe",
     "Roman philosophy", "1st c. BCE"),
    ("Prince Machiavelli", "Niccolò Machiavelli", "The Prince", "Italy",
     "Europe", "Renaissance political philosophy", "1532"),
    ("Consolation Philosophy Boethius", "Boethius",
     "The Consolation of Philosophy", "Italy", "Europe",
     "Neoplatonism", "524"),

    # ---------------- France ----------------
    ("Discourse Method Descartes", "René Descartes",
     "Discourse on the Method", "France", "Europe", "Rationalism",
     "1637"),
    ("Meditations First Philosophy Descartes", "René Descartes",
     "Meditations on First Philosophy", "France", "Europe",
     "Rationalism", "1641"),
    ("Pensees Pascal", "Blaise Pascal", "Pensées", "France", "Europe",
     "Jansenist thought", "1670"),
    ("Social Contract Rousseau", "Jean-Jacques Rousseau",
     "The Social Contract", "France", "Europe", "Enlightenment",
     "1762"),
    ("Emile Rousseau", "Jean-Jacques Rousseau", "Emile", "France",
     "Europe", "Enlightenment", "1762"),
    ("Candide Voltaire", "Voltaire", "Candide", "France", "Europe",
     "Enlightenment", "1759"),
    ("Philosophical Dictionary Voltaire", "Voltaire",
     "A Philosophical Dictionary", "France", "Europe", "Enlightenment",
     "1764"),
    ("Creative Evolution Bergson", "Henri Bergson", "Creative Evolution",
     "France", "Europe", "Philosophy of life", "1907"),

    # ---------------- Germany ----------------
    ("Critique Pure Reason Kant", "Immanuel Kant",
     "The Critique of Pure Reason", "Germany", "Europe",
     "German Idealism", "1781"),
    ("Fundamental Principles Metaphysic Morals Kant", "Immanuel Kant",
     "Fundamental Principles of the Metaphysic of Morals", "Germany",
     "Europe", "German Idealism", "1785"),
    ("Critique Practical Reason Kant", "Immanuel Kant",
     "The Critique of Practical Reason", "Germany", "Europe",
     "German Idealism", "1788"),
    ("Philosophy of History Hegel", "G. W. F. Hegel",
     "The Philosophy of History", "Germany", "Europe",
     "German Idealism", "1837"),
    ("Studies in Pessimism Schopenhauer", "Arthur Schopenhauer",
     "Studies in Pessimism", "Germany", "Europe", "Pessimism", "1851"),
    ("Wisdom of Life Schopenhauer", "Arthur Schopenhauer",
     "The Wisdom of Life", "Germany", "Europe", "Pessimism", "1851"),
    ("Thus Spake Zarathustra Nietzsche", "Friedrich Nietzsche",
     "Thus Spoke Zarathustra", "Germany", "Europe", "Existentialism",
     "1883"),
    ("Beyond Good and Evil Nietzsche", "Friedrich Nietzsche",
     "Beyond Good and Evil", "Germany", "Europe", "Existentialism",
     "1886"),
    ("Maxims Reflections Goethe", "Johann Wolfgang von Goethe",
     "Maxims and Reflections", "Germany", "Europe",
     "Weimar Classicism", "1833"),

    # ---------------- Austria ----------------
    ("General Introduction Psychoanalysis Freud", "Sigmund Freud",
     "A General Introduction to Psychoanalysis", "Austria", "Europe",
     "Psychoanalysis", "1917"),
    ("Tractatus Wittgenstein", "Ludwig Wittgenstein",
     "Tractatus Logico-Philosophicus", "Austria", "Europe",
     "Analytic philosophy", "1921"),

    # ---------------- United Kingdom ----------------
    ("Enquiry Human Understanding Hume", "David Hume",
     "An Enquiry Concerning Human Understanding", "United Kingdom",
     "Europe", "Empiricism", "1748"),
    ("Second Treatise Government Locke", "John Locke",
     "Second Treatise of Government", "United Kingdom", "Europe",
     "Empiricism", "1689"),
    ("Three Dialogues Berkeley", "George Berkeley",
     "Three Dialogues Between Hylas and Philonous", "United Kingdom",
     "Europe", "Idealism", "1713"),
    ("Leviathan Hobbes", "Thomas Hobbes", "Leviathan",
     "United Kingdom", "Europe", "Political philosophy", "1651"),
    ("Essays Francis Bacon", "Francis Bacon", "The Essays",
     "United Kingdom", "Europe", "Empiricism", "1597"),
    ("On Liberty Mill", "John Stuart Mill", "On Liberty",
     "United Kingdom", "Europe", "Utilitarianism", "1859"),
    ("Utilitarianism Mill", "John Stuart Mill", "Utilitarianism",
     "United Kingdom", "Europe", "Utilitarianism", "1863"),
    ("Problems of Philosophy Russell", "Bertrand Russell",
     "The Problems of Philosophy", "United Kingdom", "Europe",
     "Analytic philosophy", "1912"),

    # ---------------- Russia ----------------
    ("Kingdom of God Within You Tolstoy", "Leo Tolstoy",
     "The Kingdom of God Is Within You", "Russia", "Europe",
     "Christian anarchism", "1894"),
    ("Notes from Underground Dostoevsky", "Fyodor Dostoevsky",
     "Notes from Underground", "Russia", "Europe",
     "Russian existential literature", "1864"),
    ("Mutual Aid Kropotkin", "Peter Kropotkin", "Mutual Aid",
     "Russia", "Europe", "Anarchist philosophy", "1902"),

    # ---------------- Middle East ----------------
    ("Rubaiyat Omar Khayyam FitzGerald", "Omar Khayyam (tr. FitzGerald)",
     "Rubaiyat", "Iran", "Middle East & North Africa",
     "Persian poetry", "12th c."),
    ("Alchemy of Happiness Ghazali", "Al-Ghazali",
     "The Alchemy of Happiness", "Iran", "Middle East & North Africa",
     "Sufism", "11th c."),
    ("Guide for the Perplexed Maimonides", "Maimonides",
     "The Guide for the Perplexed", "Israel",
     "Middle East & North Africa", "Jewish philosophy", "12th c."),
    ("Prophet Gibran", "Khalil Gibran", "The Prophet", "Lebanon",
     "Middle East & North Africa", "Mystical humanism", "1923"),
    ("Madman Gibran", "Khalil Gibran", "The Madman", "Lebanon",
     "Middle East & North Africa", "Mystical humanism", "1918"),
    ("Forerunner Gibran", "Khalil Gibran", "The Forerunner", "Lebanon",
     "Middle East & North Africa", "Mystical humanism", "1920"),

    # ---------------- United States ----------------
    ("Walden Thoreau", "Henry David Thoreau", "Walden", "United States",
     "North America", "Transcendentalism", "1854"),
    ("Civil Disobedience Thoreau", "Henry David Thoreau",
     "On the Duty of Civil Disobedience", "United States",
     "North America", "Transcendentalism", "1849"),
    ("Essays First Series Emerson", "Ralph Waldo Emerson",
     "Essays, First Series", "United States", "North America",
     "Transcendentalism", "1841"),
    ("Nature Emerson", "Ralph Waldo Emerson", "Nature", "United States",
     "North America", "Transcendentalism", "1836"),
    ("Pragmatism William James", "William James", "Pragmatism",
     "United States", "North America", "Pragmatism", "1907"),
    ("Varieties Religious Experience James", "William James",
     "The Varieties of Religious Experience", "United States",
     "North America", "Pragmatism", "1902"),
    ("Democracy and Education Dewey", "John Dewey",
     "Democracy and Education", "United States", "North America",
     "Pragmatism", "1916"),
    ("Age of Reason Paine", "Thomas Paine", "The Age of Reason",
     "United States", "North America", "Deism", "1794"),

    # ---------------- Spain ----------------
    ("Life of Reason Santayana", "George Santayana", "The Life of Reason",
     "Spain", "Europe", "Naturalism", "1905"),

    # ============ ROUND 2 ============

    # ---------------- Greece (round 2) ----------------
    ("Gorgias Plato Jowett", "Plato (tr. Benjamin Jowett)", "Gorgias",
     "Greece", "Europe", "Ancient Greek philosophy", "4th c. BCE"),
    ("Protagoras Plato Jowett", "Plato (tr. Benjamin Jowett)",
     "Protagoras", "Greece", "Europe", "Ancient Greek philosophy",
     "4th c. BCE"),
    ("Phaedrus Plato Jowett", "Plato (tr. Benjamin Jowett)", "Phaedrus",
     "Greece", "Europe", "Ancient Greek philosophy", "4th c. BCE"),
    ("Meno Plato Jowett", "Plato (tr. Benjamin Jowett)", "Meno", "Greece",
     "Europe", "Ancient Greek philosophy", "4th c. BCE"),
    ("Cratylus Plato Jowett", "Plato (tr. Benjamin Jowett)", "Cratylus",
     "Greece", "Europe", "Ancient Greek philosophy", "4th c. BCE"),
    ("Theaetetus Plato Jowett", "Plato (tr. Benjamin Jowett)",
     "Theaetetus", "Greece", "Europe", "Ancient Greek philosophy",
     "4th c. BCE"),
    ("Timaeus Plato Jowett", "Plato (tr. Benjamin Jowett)", "Timaeus",
     "Greece", "Europe", "Ancient Greek philosophy", "4th c. BCE"),
    ("Laws Plato Jowett", "Plato (tr. Benjamin Jowett)", "Laws", "Greece",
     "Europe", "Ancient Greek philosophy", "4th c. BCE"),
    ("Statesman Plato Jowett", "Plato (tr. Benjamin Jowett)", "Statesman",
     "Greece", "Europe", "Ancient Greek philosophy", "4th c. BCE"),
    ("Sophist Plato Jowett", "Plato (tr. Benjamin Jowett)", "Sophist",
     "Greece", "Europe", "Ancient Greek philosophy", "4th c. BCE"),
    ("Charmides Plato", "Plato (tr. Benjamin Jowett)", "Charmides",
     "Greece", "Europe", "Ancient Greek philosophy", "4th c. BCE"),
    ("Lysis Plato", "Plato (tr. Benjamin Jowett)", "Lysis", "Greece",
     "Europe", "Ancient Greek philosophy", "4th c. BCE"),
    ("Euthyphro Plato", "Plato (tr. Benjamin Jowett)", "Euthyphro",
     "Greece", "Europe", "Ancient Greek philosophy", "4th c. BCE"),
    ("Ion Plato", "Plato (tr. Benjamin Jowett)", "Ion", "Greece",
     "Europe", "Ancient Greek philosophy", "4th c. BCE"),
    ("Poetics Aristotle", "Aristotle", "Poetics", "Greece", "Europe",
     "Ancient Greek philosophy", "4th c. BCE"),
    ("Metaphysics Aristotle", "Aristotle", "Metaphysics", "Greece",
     "Europe", "Ancient Greek philosophy", "4th c. BCE"),
    ("On the Soul Aristotle", "Aristotle", "On the Soul", "Greece",
     "Europe", "Ancient Greek philosophy", "4th c. BCE"),
    ("Rhetoric Aristotle", "Aristotle", "Rhetoric", "Greece", "Europe",
     "Ancient Greek philosophy", "4th c. BCE"),
    ("Discourses Epictetus", "Epictetus", "Discourses", "Greece",
     "Europe", "Stoicism", "2nd c. CE"),
    ("Lives Opinions Eminent Philosophers Laertius", "Diogenes Laertius",
     "Lives of the Eminent Philosophers", "Greece", "Europe",
     "Doxography", "3rd c. CE"),
    ("Plutarch Morals Goodwin", "Plutarch", "Plutarch's Morals",
     "Greece", "Europe", "Middle Platonism", "1st c. CE"),

    # ---------------- Egypt (round 2) ----------------
    ("Enneads Plotinus MacKenna", "Plotinus (tr. Stephen MacKenna)",
     "The Enneads", "Egypt", "Middle East & North Africa",
     "Neoplatonism", "3rd c. CE"),

    # ---------------- China (round 2) ----------------
    ("Doctrine of the Mean Legge", "Zisi (tr. James Legge)",
     "The Doctrine of the Mean", "China", "East Asia", "Confucianism",
     "5th c. BCE"),
    ("Great Learning Legge", "Zengzi (tr. James Legge)",
     "The Great Learning", "China", "East Asia", "Confucianism",
     "5th c. BCE"),
    ("Sacred Books of China Legge", "Various (tr. James Legge)",
     "The Sacred Books of China", "China", "East Asia",
     "Chinese classics", "Classical era"),
    ("Book of Changes I Ching Legge", "Unknown (tr. James Legge)",
     "The I Ching (Book of Changes)", "China", "East Asia",
     "Yijing cosmology", "c. 900 BCE"),
    ("Laotzu Tao Wu Wei Goddard", "Laozi (tr. Dwight Goddard)",
     "Laotzu's Tao and Wu Wei", "China", "East Asia", "Daoism",
     "4th c. BCE"),
    ("Han Fei Tzu", "Han Feizi", "The Complete Works of Han Fei Tzu",
     "China", "East Asia", "Legalism", "3rd c. BCE"),

    # ---------------- India (round 2) ----------------
    ("Sutta Nipata", "The Buddha (tr. V. Fausböll)", "Sutta-Nipata",
     "India", "South Asia", "Buddhism", "3rd c. BCE"),
    ("Karma Yoga Vivekananda", "Swami Vivekananda", "Karma Yoga",
     "India", "South Asia", "Vedanta", "1896"),
    ("Jnana Yoga Vivekananda", "Swami Vivekananda", "Jnana Yoga",
     "India", "South Asia", "Vedanta", "1899"),
    ("Colombo Almora Vivekananda", "Swami Vivekananda",
     "Lectures from Colombo to Almora", "India", "South Asia",
     "Vedanta", "1897"),
    ("Creative Unity Tagore", "Rabindranath Tagore", "Creative Unity",
     "India", "South Asia", "Bengali humanism", "1922"),
    ("Yoga Sutras Patanjali", "Patanjali", "The Yoga Sutras",
     "India", "South Asia", "Yoga philosophy", "c. 200 CE"),

    # ---------------- Japan (round 2) ----------------
    ("Book of Tea Okakura", "Okakura Kakuzō", "The Book of Tea",
     "Japan", "East Asia", "Japanese aesthetics", "1906"),
    ("Kokoro Hearn", "Lafcadio Hearn", "Kokoro", "Japan", "East Asia",
     "Japanese culture studies", "1896"),
    ("Glimpses Unfamiliar Japan Hearn", "Lafcadio Hearn",
     "Glimpses of Unfamiliar Japan", "Japan", "East Asia",
     "Japanese culture studies", "1894"),

    # ---------------- Italy (round 2) ----------------
    ("Discourses Livy Machiavelli", "Niccolò Machiavelli",
     "Discourses on Livy", "Italy", "Europe",
     "Renaissance political philosophy", "1531"),
    ("Summa Theologica Aquinas", "Thomas Aquinas", "Summa Theologica",
     "Italy", "Europe", "Scholasticism", "1274"),
    ("New Science Vico", "Giambattista Vico", "The New Science",
     "Italy", "Europe", "Philosophy of history", "1725"),
    ("On Duties Cicero", "Cicero", "On Duties (De Officiis)", "Italy",
     "Europe", "Roman philosophy", "44 BCE"),
    ("Letters from a Stoic Seneca", "Seneca", "Letters from a Stoic",
     "Italy", "Europe", "Stoicism", "1st c. CE"),

    # ---------------- France (round 2) ----------------
    ("Essays Montaigne Hazlitt", "Michel de Montaigne", "Essays",
     "France", "Europe", "Renaissance humanism", "1580"),
    ("Maxims La Rochefoucauld", "François de La Rochefoucauld",
     "Reflections and Moral Maxims", "France", "Europe",
     "French moralism", "1665"),
    ("Time and Free Will Bergson", "Henri Bergson", "Time and Free Will",
     "France", "Europe", "Philosophy of life", "1889"),
    ("Laughter Bergson", "Henri Bergson",
     "Laughter: An Essay on the Meaning of the Comic", "France",
     "Europe", "Philosophy of life", "1900"),

    # ---------------- Germany (round 2) ----------------
    ("Prolegomena Kant", "Immanuel Kant",
     "Prolegomena to Any Future Metaphysics", "Germany", "Europe",
     "German Idealism", "1783"),
    ("Critique of Judgment Kant", "Immanuel Kant",
     "The Critique of Judgment", "Germany", "Europe", "German Idealism",
     "1790"),
    ("Vocation of Man Fichte", "Johann Gottlieb Fichte",
     "The Vocation of Man", "Germany", "Europe", "German Idealism",
     "1800"),
    ("Human All Too Human Nietzsche", "Friedrich Nietzsche",
     "Human, All Too Human", "Germany", "Europe", "Existentialism",
     "1878"),
    ("Joyful Wisdom Nietzsche", "Friedrich Nietzsche",
     "The Gay Science", "Germany", "Europe", "Existentialism", "1882"),
    ("Genealogy of Morals Nietzsche", "Friedrich Nietzsche",
     "On the Genealogy of Morals", "Germany", "Europe",
     "Existentialism", "1887"),
    ("Twilight Idols Nietzsche", "Friedrich Nietzsche",
     "Twilight of the Idols", "Germany", "Europe", "Existentialism",
     "1889"),
    ("Ecce Homo Nietzsche", "Friedrich Nietzsche", "Ecce Homo",
     "Germany", "Europe", "Existentialism", "1888"),
    ("World as Will and Idea Schopenhauer", "Arthur Schopenhauer",
     "The World as Will and Representation", "Germany", "Europe",
     "Pessimism", "1818"),
    ("Essays of Schopenhauer", "Arthur Schopenhauer", "Essays",
     "Germany", "Europe", "Pessimism", "1851"),

    # ---------------- Austria (round 2) ----------------
    ("Interpretation of Dreams Freud", "Sigmund Freud",
     "The Interpretation of Dreams", "Austria", "Europe",
     "Psychoanalysis", "1899"),
    ("Psychopathology Everyday Life Freud", "Sigmund Freud",
     "Psychopathology of Everyday Life", "Austria", "Europe",
     "Psychoanalysis", "1901"),
    ("Totem and Taboo Freud", "Sigmund Freud", "Totem and Taboo",
     "Austria", "Europe", "Psychoanalysis", "1913"),

    # ---------------- United Kingdom (round 2) ----------------
    ("Essay Concerning Human Understanding Locke", "John Locke",
     "An Essay Concerning Human Understanding", "United Kingdom",
     "Europe", "Empiricism", "1689"),
    ("Letter Concerning Toleration Locke", "John Locke",
     "A Letter Concerning Toleration", "United Kingdom", "Europe",
     "Empiricism", "1689"),
    ("Enquiry Principles of Morals Hume", "David Hume",
     "An Enquiry Concerning the Principles of Morals", "United Kingdom",
     "Europe", "Empiricism", "1751"),
    ("Dialogues Natural Religion Hume", "David Hume",
     "Dialogues Concerning Natural Religion", "United Kingdom",
     "Europe", "Empiricism", "1779"),
    ("Treatise Human Nature Hume", "David Hume",
     "A Treatise of Human Nature", "United Kingdom", "Europe",
     "Empiricism", "1739"),
    ("Theory of Moral Sentiments Smith", "Adam Smith",
     "The Theory of Moral Sentiments", "United Kingdom", "Europe",
     "Scottish Enlightenment", "1759"),
    ("Wealth of Nations Smith", "Adam Smith",
     "An Inquiry into the Nature and Causes of the Wealth of Nations",
     "United Kingdom", "Europe", "Scottish Enlightenment", "1776"),
    ("Principles Morals Legislation Bentham", "Jeremy Bentham",
     "An Introduction to the Principles of Morals and Legislation",
     "United Kingdom", "Europe", "Utilitarianism", "1789"),
    ("Data of Ethics Spencer", "Herbert Spencer", "The Data of Ethics",
     "United Kingdom", "Europe", "Evolutionary ethics", "1879"),
    ("Principles Human Knowledge Berkeley", "George Berkeley",
     "A Treatise Concerning the Principles of Human Knowledge",
     "United Kingdom", "Europe", "Idealism", "1710"),
    ("Analysis of Mind Russell", "Bertrand Russell",
     "The Analysis of Mind", "United Kingdom", "Europe",
     "Analytic philosophy", "1921"),
    ("Mysticism and Logic Russell", "Bertrand Russell",
     "Mysticism and Logic", "United Kingdom", "Europe",
     "Analytic philosophy", "1918"),
    ("Subjection of Women Mill", "John Stuart Mill",
     "The Subjection of Women", "United Kingdom", "Europe",
     "Utilitarianism", "1869"),
    ("Sartor Resartus Carlyle", "Thomas Carlyle", "Sartor Resartus",
     "United Kingdom", "Europe", "Victorian philosophy", "1836"),
    ("Religio Medici Browne", "Thomas Browne", "Religio Medici",
     "United Kingdom", "Europe", "English Renaissance", "1643"),
    ("Evolution Ethics Huxley", "Thomas Henry Huxley",
     "Evolution and Ethics", "United Kingdom", "Europe",
     "Scientific humanism", "1893"),
    ("Novum Organum Bacon", "Francis Bacon", "Novum Organum",
     "United Kingdom", "Europe", "Empiricism", "1620"),

    # ---------------- Ireland (round 2) ----------------
    ("Soul of Man under Socialism Wilde", "Oscar Wilde",
     "The Soul of Man under Socialism", "Ireland", "Europe",
     "Aestheticism", "1891"),

    # ---------------- Russia (round 2) ----------------
    ("What Is Art Tolstoy", "Leo Tolstoy", "What Is Art?", "Russia",
     "Europe", "Christian anarchism", "1897"),
    ("My Religion Tolstoy", "Leo Tolstoy", "What I Believe", "Russia",
     "Europe", "Christian anarchism", "1884"),
    ("Confession Tolstoy", "Leo Tolstoy", "A Confession", "Russia",
     "Europe", "Christian anarchism", "1882"),
    ("Conquest of Bread Kropotkin", "Peter Kropotkin",
     "The Conquest of Bread", "Russia", "Europe",
     "Anarchist philosophy", "1892"),

    # ---------------- United States (round 2) ----------------
    ("Essays Second Series Emerson", "Ralph Waldo Emerson",
     "Essays, Second Series", "United States", "North America",
     "Transcendentalism", "1844"),
    ("Conduct of Life Emerson", "Ralph Waldo Emerson",
     "The Conduct of Life", "United States", "North America",
     "Transcendentalism", "1860"),
    ("Representative Men Emerson", "Ralph Waldo Emerson",
     "Representative Men", "United States", "North America",
     "Transcendentalism", "1850"),
    ("Talks to Teachers James", "William James",
     "Talks to Teachers on Psychology", "United States",
     "North America", "Pragmatism", "1899"),
    ("Psychology Briefer Course James", "William James",
     "Psychology: The Briefer Course", "United States", "North America",
     "Pragmatism", "1892"),
    ("How We Think Dewey", "John Dewey", "How We Think",
     "United States", "North America", "Pragmatism", "1910"),
    ("Human Nature Conduct Dewey", "John Dewey",
     "Human Nature and Conduct", "United States", "North America",
     "Pragmatism", "1922"),
    ("Chance Love Logic Peirce", "Charles Sanders Peirce",
     "Chance, Love, and Logic", "United States", "North America",
     "Pragmatism", "1923"),

    # ---------------- Lebanon (round 2) ----------------
    ("Broken Wings Gibran", "Khalil Gibran", "The Broken Wings",
     "Lebanon", "Middle East & North Africa", "Mystical humanism",
     "1912"),
    ("Jesus Son of Man Gibran", "Khalil Gibran", "Jesus the Son of Man",
     "Lebanon", "Middle East & North Africa", "Mystical humanism",
     "1928"),
    ("Spirits Rebellious Gibran", "Khalil Gibran", "Spirits Rebellious",
     "Lebanon", "Middle East & North Africa", "Mystical humanism",
     "1908"),
    ("Sand and Foam Gibran", "Khalil Gibran", "Sand and Foam",
     "Lebanon", "Middle East & North Africa", "Mystical humanism",
     "1926"),

    # ---------------- Iran (round 2) ----------------
    ("Gulistan Saadi", "Saadi Shirazi", "The Gulistan", "Iran",
     "Middle East & North Africa", "Persian ethics", "1258"),

    # ---------------- Israel (round 2) ----------------
    ("Ethics of the Fathers Taylor", "Various (tr. Charles Taylor)",
     "Sayings of the Jewish Fathers (Pirkei Avot)", "Israel",
     "Middle East & North Africa", "Jewish ethics", "2nd c. CE"),

    # ---------------- Sweden (round 2) ----------------
    ("Heaven and Hell Swedenborg", "Emanuel Swedenborg",
     "Heaven and Hell", "Sweden", "Europe", "Swedenborgianism",
     "1758"),
    ("Divine Love and Wisdom Swedenborg", "Emanuel Swedenborg",
     "Angelic Wisdom Concerning the Divine Love and Wisdom", "Sweden",
     "Europe", "Swedenborgianism", "1763"),

    # ---------------- Netherlands (round 2) ----------------
    ("Theologico-Political Treatise Spinoza", "Baruch Spinoza",
     "Theologico-Political Treatise", "Netherlands", "Europe",
     "Rationalism", "1670"),
    ("Improvement Understanding Spinoza", "Baruch Spinoza",
     "On the Improvement of the Understanding", "Netherlands", "Europe",
     "Rationalism", "1677"),
]


def http_get(url, timeout=90, retries=3):
    last_exc = None

    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": USER_AGENT}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:
            last_exc = exc
            wait = min(2 ** attempt * 2, 20)
            print(f"  request failed ({exc}); retry {attempt}/{retries} "
                  f"in {wait}s…")
            time.sleep(wait)

    raise last_exc


def find_book(query):
    """Search Gutendex; return (title, txt_url) of the best match."""
    url = GUTENDEX + "?" + urllib.parse.urlencode({"search": query})
    data = json.loads(http_get(url))

    for book in data.get("results", []):
        formats = book.get("formats", {})
        for mime, link in formats.items():
            if mime.startswith("text/plain") and link.endswith(".txt"):
                return book.get("title", query), link

    return None, None


def strip_gutenberg(raw):
    """Remove the Project Gutenberg header/footer boilerplate."""
    text = raw.decode("utf-8", errors="replace")

    start = re.search(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG",
                      text)
    end = re.search(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG", text)

    if start:
        text = text[start.end():]
        text = text[text.find("\n"):]
    if end:
        text = text[: end.start()]

    return text


def chunk_book(text):
    """Split a book into 120–250 word passages at paragraph breaks."""
    text = re.sub(r"\r\n?", "\n", text)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text)]

    passages = []
    current, words = [], 0

    for p in paragraphs:
        p = re.sub(r"\s+", " ", p)
        w = len(p.split())

        # skip headings, TOCs, tiny fragments
        if w < 15 or (p.isupper() and w < 12):
            continue
        if "gutenberg" in p.lower()[:40]:
            continue

        current.append(p)
        words += w

        if words >= PASSAGE_MIN_WORDS:
            passage = " ".join(current)
            if len(passage.split()) <= PASSAGE_MAX_WORDS * 1.5:
                passages.append(passage)
            current, words = [], 0

        if len(passages) >= MAX_PASSAGES_PER_BOOK:
            break

    return passages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", action="store_true",
                        help="Print corpus statistics and exit.")
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)

    if args.stats:
        total = conn.execute("SELECT COUNT(*) FROM texts").fetchone()[0]
        print(f"Texts: {total}")
        for row in conn.execute(
            "SELECT country, COUNT(*) c FROM texts GROUP BY country"
            " ORDER BY c DESC LIMIT 15"
        ):
            print(f"  {row[0]}: {row[1]}")
        return

    existing = {
        row[0] for row in conn.execute("SELECT text FROM texts")
    }

    country_counts = dict(
        conn.execute(
            "SELECT country, COUNT(*) FROM texts GROUP BY country"
        ).fetchall()
    )

    total_added = 0

    for (query, author, work, country, region, tradition, era) in BOOKS:
        if country_counts.get(country, 0) >= TARGET_PER_COUNTRY:
            print(f"{country}: target {TARGET_PER_COUNTRY} reached, "
                  f"skipping {work}")
            continue

        print(f"\n=== {work} — {author} ({country}) ===")

        try:
            title, txt_url = find_book(query)
        except Exception as exc:
            print(f"  search failed: {exc}")
            continue

        if not txt_url:
            print("  no plain-text edition found; skipping")
            continue

        print(f"  downloading: {title}")
        time.sleep(REQUEST_PAUSE)

        try:
            raw = http_get(txt_url)
        except Exception as exc:
            print(f"  download failed: {exc}")
            continue

        passages = chunk_book(strip_gutenberg(raw))
        print(f"  {len(passages)} passages extracted")

        rows = []
        for p in passages:
            if p in existing:
                continue
            existing.add(p)
            rows.append((
                p, author, work, country, region, tradition, "English",
                era, "Project Gutenberg (public domain)",
            ))
            if country_counts.get(country, 0) + len(rows) \
                    >= TARGET_PER_COUNTRY:
                break

        if rows:
            conn.executemany(
                "INSERT INTO texts (text, author, work, country, region,"
                " tradition, language, era, source)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                rows,
            )
            conn.commit()

        country_counts[country] = country_counts.get(country, 0) + len(rows)
        total_added += len(rows)
        print(f"  inserted {len(rows)} "
              f"(country now {country_counts[country]})")

    total = conn.execute("SELECT COUNT(*) FROM texts").fetchone()[0]
    countries = conn.execute(
        "SELECT COUNT(DISTINCT country) FROM texts").fetchone()[0]
    print(f"\nDone. Added {total_added} passages.")
    print(f"Corpus now: {total} texts from {countries} countries.")
    print("Reopen the app to embed the new texts (first run will be "
          "slow for thousands of passages, then cached).")

    conn.close()


if __name__ == "__main__":
    main()
