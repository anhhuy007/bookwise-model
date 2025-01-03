import os
import psycopg2
from psycopg2 import pool
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the database connection string from the environment variable
DATABASE_URL = os.getenv("DATABASE_URL")

# Create a connection pool (initialize to None)
connection_pool = None

def create_connection_pool():
    """Creates a connection pool if it doesn't exist."""
    global connection_pool
    if connection_pool is None:
        try:
            connection_pool = pool.SimpleConnectionPool(1, 10, DATABASE_URL)
            print("Connection pool created successfully")
        except (Exception, psycopg2.DatabaseError) as error:
            print("Error while creating connection pool", error)


def get_connection():
    """Gets a connection from the pool."""
    create_connection_pool()  # Ensure the pool is created
    if connection_pool:
        return connection_pool.getconn()
    else:
        print("Failed to get connection: Connection pool not initialized.")
        return None


def release_connection(conn):
    """Returns a connection to the pool."""
    if connection_pool:
        connection_pool.putconn(conn)
    else:
        print("Failed to release connection: Connection pool not initialized.")


def close_all_connections():
    """Closes all connections in the pool."""
    if connection_pool:
        connection_pool.closeall()
        print("Connection pool closed")
    else:
        print("Failed to close connections: Connection pool not initialized.")


# Function to fetch data from the 'books' table
def get_books_data():
    conn = None
    try:
        conn = get_connection()
        if conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM books")
                books = cur.fetchall()
                return books
        else:
            print("Failed to get connection for fetching books data.")
            return []
    except (Exception, psycopg2.Error) as error:
        print("Error fetching data from books table", error)
        return []
    finally:
        if conn:
            release_connection(conn)

def get_books_summarize_data():
    conn = None
    try:
        conn = get_connection()
        if conn:
            with conn.cursor() as cur:
                cur.execute("SELECT title, authors, description FROM books")
                books = cur.fetchall()
                return books
        else:
            print("Failed to get connection for fetching books data.")
            return []
    except (Exception, psycopg2.Error) as error:
        print("Error fetching data from books table", error)
        return []
    finally:
        if conn:
            release_connection(conn)

# Function to fetch data from the 'user_info' table
def get_user_info_data():
    conn = None
    try:
        conn = get_connection()
        if conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM user_info")
                user_info = cur.fetchall()
                return user_info
        else:
            print("Failed to get connection for fetching user_info data.")
            return []
    except (Exception, psycopg2.Error) as error:
        print("Error fetching data from user_info table", error)
        return []
    finally:
        if conn:
            release_connection(conn)


# Function to fetch data from the 'favourite_books' table
def get_favourite_books_data():
    conn = None
    try:
        conn = get_connection()
        if conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM favourite_books")
                favourite_books = cur.fetchall()
                return favourite_books
        else:
            print("Failed to get connection for fetching favourite_books data.")
            return []
    except (Exception, psycopg2.Error) as error:
        print("Error fetching data from favourite_books table", error)
        return []
    finally:
        if conn:
            release_connection(conn)
            
def get_book_by_id(id: int):
    conn = None
    try:
        conn = get_connection()
        if conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM books WHERE id = %s", (id,))
                book = cur.fetchone()
                return book
        else:
            print("Failed to get connection for fetching book by id.")
            return None
    except (Exception, psycopg2.Error) as error:
        print("Error fetching book by id", error)
        return None
    finally:
        if conn:
            release_connection(conn)


# Example usage in your main application logic:
def fetch_all_data():
    books_data = get_books_data()
    user_info_data = get_user_info_data()
    favourite_books_data = get_favourite_books_data()

    print("Books Data:", len(books_data), "records")
    print("User Info Data:", len(user_info_data), "records")
    print("Favourite Books Data:", len(favourite_books_data), "records")
    
    return books_data, user_info_data, favourite_books_data
