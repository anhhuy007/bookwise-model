CREATE TABLE users (
  id SERIAL PRIMARY KEY,
  username VARCHAR(50) UNIQUE NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  email VARCHAR(50) UNIQUE NOT NULL,
  avatar_url VARCHAR(255) NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE books (
    id SERIAL PRIMARY KEY,
    isbn13 VARCHAR(13) UNIQUE NOT NULL,
    title VARCHAR(255) NOT NULL,
    authors VARCHAR(255) NOT NULL, 
    published_date VARCHAR(10),
    page_count INTEGER DEFAULT 0, 
    category VARCHAR DEFAULT 'Other',
    language VARCHAR(2) NOT NULL,
    avg_rating FLOAT DEFAULT 0.0,
    rating_count INTEGER DEFAULT 0,
    img_url VARCHAR NOT NULL, 
    preview_url VARCHAR, 
    description TEXT
);

CREATE TABLE categories (
  name VARCHAR(255) PRIMARY KEY, 
  description TEXT,
  view_count INTEGER DEFAULT 0
);

CREATE TABLE ratings (
  book_id INTEGER NOT NULL,
  user_id INTEGER NOT NULL,
  rating FLOAT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY (book_id, user_id),
  FOREIGN KEY (book_id) REFERENCES books(id) ON DELETE CASCADE,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE favourite_books (
  user_id INTEGER NOT NULL,
  book_id INTEGER NOT NULL,
  added_at TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY (user_id, book_id),
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
  FOREIGN KEY (book_id) REFERENCES books(id) ON DELETE CASCADE
);

CREATE TABLE authors (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    view_count INTEGER DEFAULT 0,
    avatar_url VARCHAR,
    description TEXT
);

CREATE TABLE user_info (
  id INTEGER PRIMARY KEY,
  gender BOOLEAN,
  dob DATE,
  university VARCHAR(150),
  faculty VARCHAR(100),
  age INTEGER,
  language VARCHAR(2),
  factor INTEGER,
  goal VARCHAR(20)
);

CREATE TABLE following_authors (
  user_id INTEGER NOT NULL,
  author_id INTEGER NOT NULL,
  added_at TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY (user_id, author_id),
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
  FOREIGN KEY (author_id) REFERENCES authors(id) ON DELETE CASCADE
);

