import os
import psycopg2
import requests
import json
import re
import numpy as np
import cv2
import pytesseract
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory, render_template
from werkzeug.utils import secure_filename
from flask_cors import CORS
from gtts import gTTS
import io
import base64
from dotenv import load_dotenv
import datetime
import jwt
from functools import wraps

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__, template_folder='templates')
CORS(app, origins=["http://localhost:8081", "http://localhost:8082"], supports_credentials=True) # Allow both web and mobile dev servers

@app.before_request
def log_request_info():
    # Log all requests for debugging
    print(f"{request.method} {request.path}")
    

@app.route('/')
def index():
    return jsonify({'message': 'Welcome to the Pet Tracker API!'})

@app.route('/uploads/<path:path>')
def send_upload(path):
    return send_from_directory('uploads', path)

@app.route('/card/<int:pet_id>')
def pet_card(pet_id):
    """Renders a public-facing card with pet details."""
    conn = get_db_connection()
    if conn is None:
        return "Database connection failed.", 500
    
    pet = None
    owner = None
    
    try:
        with conn.cursor() as cur:
            # Fetch pet details along with owner's name
            cur.execute("""
                SELECT a.id, a.name, a.species, a.breed, a.photo_url, u.full_name
                FROM animals a
                JOIN users u ON a.user_id = u.id
                WHERE a.id = %s
            """, (pet_id,))
            
            record = cur.fetchone()
            
            if record:
                base_url = request.host_url.rstrip('/')
                # Use the photo_url from the DB, or a placeholder if it's missing
                relative_photo_url = record[4] if record[4] else '/uploads/pets/placeholder.svg'
                
                # Construct an absolute URL if it's not already one
                absolute_photo_url = relative_photo_url
                if not relative_photo_url.startswith('http'):
                    absolute_photo_url = f"{base_url}{relative_photo_url}"

                pet = {
                    'id': record[0],
                    'name': record[1],
                    'species': record[2],
                    'breed': record[3],
                    'photo_url': absolute_photo_url
                }
                owner = {
                    'full_name': record[5]
                }
            else:
                return "Pet not found.", 404

    except Exception as e:
        print(f"Error fetching pet card data: {e}")
        return "An error occurred.", 500
    finally:
        if conn:
            conn.close()
            
    return render_template('card.html', pet=pet, owner=owner)


# Environment variables
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
UPLOAD_FOLDER = 'uploads/pets'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def get_db_connection():
    """Establishes a connection to the NeonDB database using the connection URL."""
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable not set.")
    conn = psycopg2.connect(DATABASE_URL)
    return conn

def get_user_by_cin(cin):
    """Fetches a user from the database by their CIN."""
    conn = get_db_connection()
    if conn is None:
        return None, "Database connection failed."
    user_data = None
    db_error = None
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, cin, full_name FROM users WHERE cin = %s;", (cin,))
            user = cur.fetchone()
            if user:
                user_data = {'id': user[0], 'cin': user[1], 'full_name': user[2]}
    except Exception as e:
        db_error = str(e)
    finally:
        if conn:
            conn.close()
    return user_data, db_error

def get_pets_by_user_id(user_id):
    """Fetches all pets for a given user ID from the database."""
    conn = get_db_connection()
    if conn is None: return None, "Database connection failed."
    pets_data = []
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, name, species, breed, date_of_birth, photo_url FROM animals WHERE user_id = %s;", (user_id,))
            pets = cur.fetchall()
            for pet in pets:
                pets_data.append({
                    'id': pet[0],
                    'name': pet[1],
                    'species': pet[2],
                    'breed': pet[3],
                    'date_of_birth': pet[4].isoformat() if pet[4] else None,
                    'photo_url': pet[5]
                })
            return pets_data, None
    except Exception as e:
        return None, str(e)
    finally:
        if conn:
            conn.close()

def get_user_and_pet_data(user_id):
    """Fetches all data for a given user ID from the database."""
    conn = get_db_connection()
    if conn is None:
        return None, None
    
    try:
        with conn.cursor() as cur:
            # Fetch user data
            cur.execute("SELECT cin, full_name, created_at FROM users WHERE id = %s", (user_id,))
            user_data = cur.fetchone()
            
            # Fetch pet data
            cur.execute("SELECT name, species, breed, date_of_birth FROM animals WHERE user_id = %s", (user_id,))
            pet_data = cur.fetchall()
            
            return user_data, pet_data
    except Exception as e:
        print(f"Database error in get_user_and_pet_data: {e}")
        return None, None
    finally:
        if conn:
            conn.close()

def create_locations_table():
    """Creates the locations table if it does not exist."""
    conn = get_db_connection()
    if conn is None:
        print("Database connection failed, could not create locations table.")
        return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS locations (
                    id SERIAL PRIMARY KEY,
                    pet_id INTEGER NOT NULL,
                     DECIMAL(9, 6) NOT NULL,
                    longitude DECIMAL(9, 6) NOT NULL,
                    timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (pet_id) REFERENCES animals (id) ON DELETE CASCADE
                );
            """)
            conn.commit()
            print("Locations table checked/created successfully.")
    except Exception as e:
        print(f"Error creating locations table: {e}")
    finally:
        if conn:
            conn.close()

# Call this function at startup to ensure the table exists
create_locations_table()

@app.route('/pet/<int:pet_id>/location', methods=['POST'])
def add_pet_location(pet_id):
    """Adds a new location for a pet."""
    data = request.get_json()
    if not data or 'latitude' not in data or 'longitude' not in data:
        return jsonify({'status': 'error', 'message': 'Latitude and longitude are required.'}), 400

    latitude = data['latitude']
    longitude = data['longitude']

    conn = get_db_connection()
    if conn is None:
        return jsonify({'status': 'error', 'message': 'Database connection failed.'}), 500

    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO locations (pet_id, latitude, longitude) VALUES (%s, %s, %s)",
                (pet_id, latitude, longitude)
            )
            conn.commit()
        return jsonify({'status': 'success', 'message': 'Location added successfully.'}), 201
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        if conn:
            conn.close()

@app.route('/pet/<int:pet_id>/location', methods=['GET'])
def get_pet_location(pet_id):
    """Gets the latest location for a pet."""
    conn = get_db_connection()
    if conn is None:
        return jsonify({'status': 'error', 'message': 'Database connection failed.'}), 500

    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT latitude, longitude FROM locations WHERE pet_id = %s ORDER BY timestamp DESC LIMIT 1",
                (pet_id,)
            )
            location = cur.fetchone()
            if location:
                return jsonify({'latitude': float(location[0]), 'longitude': float(location[1])})
            else:
                return jsonify({'status': 'error', 'message': 'No location found for this pet.'}), 404
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        if conn:
            conn.close()

# --- Authentication ---

@app.route('/login', methods=['POST'])
def login():
    """Authenticates a user by CIN and returns user info. If CIN not found, create a new user. JWT is not returned."""
    data = request.get_json()
    if not data or 'cin' not in data:
        return jsonify({'status': 'error', 'message': 'CIN is required for login.'}), 400
    cin = data['cin']
    user, error = get_user_by_cin(cin)
    if error:
        return jsonify({'status': 'error', 'message': 'Database error: ' + error}), 500
    if not user:
        # Create new user with default name
        conn = get_db_connection()
        if conn is None:
            return jsonify({'status': 'error', 'message': 'Database connection failed when trying to create new user'}), 500
        try:
            with conn.cursor() as cur:
                default_name = f"User {cin}"
                cur.execute(
                    "INSERT INTO users (cin, full_name, created_at) VALUES (%s, %s, NOW()) RETURNING id, cin, full_name;",
                    (cin, default_name)
                )
                new_user = cur.fetchone()
                conn.commit()
                if new_user:
                    user = {'id': new_user[0], 'cin': new_user[1], 'full_name': new_user[2]}
                else:
                    return jsonify({'status': 'error', 'message': 'Failed to create user from CIN.'}), 500
        except Exception as e:
            conn.rollback()
            return jsonify({'status': 'error', 'message': f'Error creating new user: {str(e)}'}), 500
        finally:
            conn.close()
    # No JWT, just return user info
    return jsonify({'status': 'success', 'user': user})

def get_current_user_id_from_token():
    """Helper to decode JWT from Authorization header and get user ID."""
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        return None
    
    try:
        # Expecting "Bearer <token>"
        token = auth_header.split(" ")[1]
        decoded_token = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return decoded_token.get('user_id')
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError, IndexError):
        return None


@app.route('/chat', methods=['POST'])
def chat_handler():
    """Handles the conversational AI logic."""
    data = request.json
    user_id = data.get('userId')
    user_question = data.get('question')

    if not all([user_id, user_question, OPENROUTER_API_KEY]):
        return jsonify({'error': 'Missing data or API key.'}), 400

    try:
        user_data, pet_data = get_user_and_pet_data(user_id)
        if not user_data:
            return jsonify({'error': 'User not found.'}), 404

        # Construct the prompt
        prompt_context = f"""You are a helpful AI assistant for a user whose details are below. 
        User Info: CIN={user_data[0]}, Name={user_data[1]}, Account Created At={user_data[2]}.
        User's Pets: {json.dumps(pet_data, default=json_serial)}
        The user's question is: '{user_question}'.
        Based on all this information, provide a helpful and concise answer. 
        IMPORTANT: You MUST answer in the Arabic language."""

        # Call OpenRouter API
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            },
            json={
                "model": "mistralai/mistral-7b-instruct:free",
                "messages": [{"role": "user", "content": prompt_context}]
            }
        )
        response.raise_for_status() # Raise an exception for bad status codes
        ai_response_text = response.json()['choices'][0]['message']['content']

        # Generate speech from the AI's Arabic response
        tts = gTTS(ai_response_text, lang='ar')
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        audio_b64 = base64.b64encode(mp3_fp.read()).decode('utf-8')

        return jsonify({'audio': f"data:audio/mp3;base64,{audio_b64}"})

    except Exception as e:
        print(f"Error in /chat: {e}")
        return jsonify({'error': str(e)}), 500

# --- Image and OCR Processing ---

def preprocess_for_ocr(image_array):
    """Preprocesses an image numpy array for better OCR results."""
    if len(image_array.shape) == 2:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    elif image_array.shape[2] == 4:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

class MRZParser:
    @staticmethod
    def parse(ocr_text):
        lines = ocr_text.strip().split('\n')
        cleaned_lines = [re.sub(r'[^A-Z0-9<]', '', line.upper()) for line in lines if line.strip()]
        print(f"--- MRZ DEBUG: Cleaned OCR Input ---\n{cleaned_lines}\n---------------------------------")

        mrz_lines = [l for l in cleaned_lines if len(l) > 20 and '<' in l]
        if len(mrz_lines) < 2:
            return {"Error": "Invalid MRZ data: Not enough valid lines found."}

        full_mrz_text = "".join(mrz_lines)
        parsed_data = {}
        print(f"--- MRZ DEBUG ---\nFull Text Block: {full_mrz_text}\n-------------------")

        # Extract CIN (existing logic)
        try:
            cin_found = False
            match = re.search(r'I<MAR[A-Z0-9]{8}<[A-Z0-9]([A-Z0-9]{8})', full_mrz_text)
            if match:
                parsed_data['CIN'] = match.group(1)
                cin_found = True
                print(f"Extracted Personal Number as CIN '{parsed_data['CIN']}' with full pattern")
            if not cin_found:
                mar_match = re.search(r'I<MAR', full_mrz_text)
                if mar_match:
                    search_text = full_mrz_text[mar_match.end():]
                    all_8_char_blocks = re.findall(r'[A-Z0-9]{8}', search_text)
                    if len(all_8_char_blocks) >= 2:
                        parsed_data['CIN'] = all_8_char_blocks[1]
                        cin_found = True
                        print(f"Extracted Personal Number as CIN '{parsed_data['CIN']}' with fallback pattern")
            if not cin_found:
                fallback_match = re.search(r'([A-Z]{2}[0-9]{3,6})', full_mrz_text)
                if fallback_match:
                    parsed_data['CIN'] = fallback_match.group(1)
                    cin_found = True
                    print(f"Extracted CIN '{parsed_data['CIN']}' with fallback pattern 3")
            if not cin_found:
                return {"Error": "Could not parse CIN from MRZ."}
        except Exception as e:
            print(f"Error parsing CIN: {e}")
            return {"Error": f"Parsing failed: {e}"}

        # Extract full name from the last MRZ line (for Moroccan IDs)
        try:
            name_line = mrz_lines[-1]
            if '<<' in name_line:
                surname, given_names = name_line.split('<<', 1)
                surname = surname.replace('<', ' ').strip()
                given_names = given_names.replace('<', ' ').strip()
                parsed_data['lastname'] = surname
                parsed_data['firstname'] = given_names
                print(f"Extracted Name: '{surname} {given_names}' (Last Name: '{surname}', First Name: '{given_names}')")
        except Exception as e:
            print(f"Error parsing Name: {e}")

        return parsed_data

@app.route('/ocr', methods=['POST'])
def ocr_handler():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part in the request.'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected.'}), 400

    try:
        in_memory_file = io.BytesIO()
        file.save(in_memory_file)
        in_memory_file.seek(0)
        image = Image.open(in_memory_file)
        image_np = np.array(image)
        processed_image = preprocess_for_ocr(image_np)
        ocr_text = pytesseract.image_to_string(processed_image)
        mrz_data = MRZParser.parse(ocr_text)
        if 'Error' in mrz_data:
            return jsonify({'status': 'error', 'message': mrz_data['Error']}), 400
        if 'CIN' in mrz_data or 'cin' in mrz_data:
            cin = mrz_data.get('CIN') or mrz_data.get('cin')
            name = None
            if 'firstname' in mrz_data and 'lastname' in mrz_data:
                name = f"{mrz_data['firstname']} {mrz_data['lastname']}"
            user, db_error = get_user_by_cin(cin)
            if db_error:
                return jsonify({'status': 'error', 'message': f'Database error: {db_error}'}), 500
            if user:
                return jsonify({'status': 'success', 'user': user})
            else:
                # User not found - create a new user with extracted CIN and name
                conn = get_db_connection()
                if conn is None:
                    return jsonify({'status': 'error', 'message': 'Database connection failed when trying to create new user'}), 500
                try:
                    with conn.cursor() as cur:
                        default_name = name if name else f"User {cin}"
                        cur.execute(
                            "INSERT INTO users (cin, full_name, created_at) VALUES (%s, %s, NOW()) RETURNING id, cin, full_name;",
                            (cin, default_name)
                        )
                        new_user = cur.fetchone()
                        conn.commit()
                        if new_user:
                            new_user_data = {'id': new_user[0], 'cin': new_user[1], 'full_name': new_user[2]}
                            return jsonify({
                                'status': 'success',
                                'user': new_user_data,
                                'message': 'New user created from ID card information.'
                            })
                        else:
                            return jsonify({'status': 'error', 'message': 'Failed to create user from ID card.'}), 500
                except Exception as e:
                    conn.rollback()
                    return jsonify({'status': 'error', 'message': f'Error creating new user: {str(e)}'}), 500
                finally:
                    conn.close()
        else:
            return jsonify({'status': 'error', 'message': 'Failed to extract CIN from ID.'}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'An unexpected error occurred: {str(e)}'}), 500

@app.route('/user/<int:user_id>/pets', methods=['GET', 'POST'])
def user_pets_handler(user_id):
    """API endpoint to get all pets for a specific user or add a new one."""
    if request.method == 'GET':
        pets, db_error = get_pets_by_user_id(user_id)
        if db_error:
            return jsonify({'status': 'error', 'message': f'Database error: {db_error}'}), 500
        return jsonify({'status': 'success', 'pets': pets if pets is not None else []})

    if request.method == 'POST':
        # Check if this is a multipart/form-data request (with image) or just JSON
        if request.content_type and 'multipart/form-data' in request.content_type:
            # Handle form data with possible image upload
            name = request.form.get('name')
            species = request.form.get('species')
            breed = request.form.get('breed')
            dob = request.form.get('date_of_birth')  # Expected format: 'YYYY-MM-DD'
            image_file = request.files.get('image')
        else:
            # Handle regular JSON request
            data = request.get_json()
            if not data or not data.get('name'):
                return jsonify({'status': 'error', 'message': 'Invalid input: name is required'}), 400
            
            name = data.get('name')
            species = data.get('species')
            breed = data.get('breed')
            dob = data.get('date_of_birth')  # Expected format: 'YYYY-MM-DD'
            image_file = None

        # Validate required fields
        if not name:
            return jsonify({'status': 'error', 'message': 'Invalid input: name is required'}), 400

        conn = get_db_connection()
        if conn is None:
            return jsonify({'status': 'error', 'message': 'Database connection failed'}), 500
        
        try:
            with conn.cursor() as cur:
                photo_url = None
                
                # First insert the pet to get an ID
                cur.execute(
                    "INSERT INTO animals (user_id, name, species, breed, date_of_birth) VALUES (%s, %s, %s, %s, %s) RETURNING id;",
                    (user_id, name, species, breed, dob)
                )
                new_pet_id = cur.fetchone()[0]
                
                # If there's an image, handle it
                if image_file and image_file.filename:
                    # Create uploads directory if it doesn't exist
                    upload_dir = os.path.join(os.getcwd(), 'uploads', 'pets')
                    os.makedirs(upload_dir, exist_ok=True)
                    
                    # Generate unique filename
                    file_extension = image_file.filename.rsplit('.', 1)[1].lower() if '.' in image_file.filename else 'jpg'
                    filename = f"pet_{new_pet_id}_{int(datetime.datetime.now().timestamp())}.{file_extension}"
                    file_path = os.path.join(upload_dir, filename)
                    
                    # Save file
                    image_file.save(file_path)
                    
                    # Update the database with the photo URL
                    photo_url = f"/uploads/pets/{filename}"
                    cur.execute(
                        "UPDATE animals SET photo_url = %s WHERE id = %s;",
                        (photo_url, new_pet_id)
                    )
                
                conn.commit()

                # Fetch the newly created pet to return it in the response
                cur.execute("SELECT id, name, species, breed, date_of_birth, photo_url FROM animals WHERE id = %s;", (new_pet_id,))
                new_pet = cur.fetchone()
                pet_data = {
                    'id': new_pet[0],
                    'name': new_pet[1],
                    'species': new_pet[2],
                    'breed': new_pet[3],
                    'date_of_birth': new_pet[4].isoformat() if new_pet[4] else None,
                    'photo_url': new_pet[5]
                }
                return jsonify({'status': 'success', 'message': 'Pet added successfully', 'pet': pet_data}), 201
        except Exception as e:
            conn.rollback()
            return jsonify({'status': 'error', 'message': f'Database error: {str(e)}'}), 500
        finally:
            if conn:
                conn.close()


@app.route('/pet/<int:pet_id>', methods=['GET', 'PUT', 'DELETE'])
def pet_handler(pet_id):
    """Handles operations for a single pet: fetching, updating, and deleting."""
    print(f"Pet handler called for pet_id: {pet_id}, method: {request.method}")
    conn = get_db_connection()
    if conn is None:
        return jsonify({'status': 'error', 'message': 'Database connection failed'}), 500

    try:
        with conn.cursor() as cur:
            if request.method == 'GET':
                # Allow GET without authentication
                cur.execute("SELECT id, name, species, breed, date_of_birth, user_id, photo_url FROM animals WHERE id = %s;", (pet_id,))
                pet_data = cur.fetchone()
                if not pet_data:
                    return jsonify({'status': 'error', 'message': 'Pet not found'}), 404
                    
                pet_json = {
                    'id': pet_data[0],
                    'name': pet_data[1],
                    'species': pet_data[2],
                    'breed': pet_data[3],
                    'date_of_birth': pet_data[4].isoformat() if pet_data[4] else None,
                    'user_id': pet_data[5],
                    'photo_url': pet_data[6]
                }
                return jsonify({'status': 'success', 'pet': pet_json})
            
            # Check if pet exists for PUT and DELETE
            cur.execute("SELECT user_id FROM animals WHERE id = %s;", (pet_id,))
            pet = cur.fetchone()
            if not pet:
                return jsonify({'status': 'error', 'message': 'Pet not found'}), 404

            elif request.method == 'PUT':
                data = request.get_json()
                if not data:
                    return jsonify({'status': 'error', 'message': 'Invalid input'}), 400

                update_fields = []
                update_values = []
                for field in ['name', 'species', 'breed', 'date_of_birth']:
                    if field in data:
                        update_fields.append(f"{field} = %s")
                        update_values.append(data[field])
                
                if not update_fields:
                    return jsonify({'status': 'error', 'message': 'No fields to update'}), 400

                update_values.append(pet_id)

                query = f"UPDATE animals SET {', '.join(update_fields)} WHERE id = %s RETURNING id, name, species, breed, date_of_birth, photo_url;"
                
                cur.execute(query, tuple(update_values))
                updated_pet = cur.fetchone()
                conn.commit()

                pet_json = {
                    'id': updated_pet[0],
                    'name': updated_pet[1],
                    'species': updated_pet[2],
                    'breed': updated_pet[3],
                    'date_of_birth': updated_pet[4].isoformat() if updated_pet[4] else None,
                    'photo_url': updated_pet[5]
                }
                return jsonify({'status': 'success', 'message': 'Pet updated successfully', 'pet': pet_json})

            elif request.method == 'DELETE':
                cur.execute("DELETE FROM animals WHERE id = %s;", (pet_id,))
                conn.commit()
                if cur.rowcount > 0:
                    return jsonify({'status': 'success', 'message': 'Pet deleted successfully'})
                else:
                    return jsonify({'status': 'error', 'message': 'Pet could not be deleted or was already deleted.'}), 404

    except Exception as e:
        conn.rollback()
        return jsonify({'status': 'error', 'message': f'Database error: {str(e)}'}), 500
    finally:
        if conn:
            conn.close()

@app.route('/animal/<int:animal_id>')
def animal_card(animal_id):
    """Display animal information in a web page format for NFC cards."""
    # Add a query parameter to force cache busting when needed
    cache_buster = request.args.get('cache', None)
    
    conn = get_db_connection()
    if conn is None:
        return f"<html><body><h1>Database connection failed</h1></body></html>", 500

    try:
        with conn.cursor() as cur:
            # Get animal details with owner information
            cur.execute("""
                SELECT a.id, a.name, a.species, a.breed, a.date_of_birth, a.photo_url, a.created_at,
                       u.full_name, u.cin
                FROM animals a 
                JOIN users u ON a.user_id = u.id 
                WHERE a.id = %s;
            """, (animal_id,))
            
            animal_data = cur.fetchone()
            
            if not animal_data:
                return f"""
                <html>
                <head>
                    <title>Pet Not Found</title>
                    <meta charset="UTF-8">
                    <style>
                        body {{ font-family: Arial, sans-serif; text-align: center; padding: 50px; }}
                        .container {{ max-width: 600px; margin: 0 auto; }}
                        .error {{ color: #e74c3c; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1 class="error">Pet Not Found</h1>
                        <p>The pet with ID {animal_id} was not found in our database.</p>
                    </div>
                </body>
                </html>
                """, 404

            # Extract data
            pet_id, name, species, breed, dob, photo_url, created_at, owner_name, owner_cin = animal_data
            
            # Format date of birth
            dob_formatted = dob.strftime('%Y-%m-%d') if dob else 'Not specified'
            created_formatted = created_at.strftime('%Y-%m-%d %H:%M') if created_at else 'Not specified'
            
            # Add cache-busting parameter to photo URL
            if photo_url:
                photo_url = f"{photo_url}?t={int(datetime.datetime.now().timestamp())}"
            
            # Default image if no photo - use a data URL for the placeholder instead of a file
            if not photo_url:
                # Simple SVG data URL
                photo_url = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTUwIiBoZWlnaHQ9IjE1MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTUwIiBoZWlnaHQ9IjE1MCIgZmlsbD0iI2NjY2NjYyIvPjx0ZXh0IHg9IjUwJSIgeT0iNTAlIiBmb250LWZhbWlseT0iQXJpYWwiIGZvbnQtc2l6ZT0iMTYiIGZpbGw9IiM2NjY2NjYiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGFsaWdubWVudC1iYXNlbGluZT0ibWlkZGxlIj5ObyBQaG90bzwvdGV4dD48L3N2Zz4='
            
            # Generate HTML page
            html_content = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>{name} - Pet Profile</title>
                <style>
                    body {{
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        margin: 0;
                        padding: 20px;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        min-height: 100vh;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                    }}
                    .card {{
                        background: white;
                        border-radius: 20px;
                        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                        overflow: hidden;
                        max-width: 500px;
                        width: 100%;
                    }}
                    .header {{
                        background: linear-gradient(45deg, #4CAF50, #45a049);
                        color: white;
                        padding: 30px;
                        text-align: center;
                    }}
                    .pet-photo {{
                        width: 150px;
                        height: 150px;
                        border-radius: 50%;
                        object-fit: cover;
                        border: 5px solid white;
                        margin-bottom: 20px;
                    }}
                    .pet-name {{
                        font-size: 2.5em;
                        margin: 0;
                        font-weight: bold;
                    }}
                    .pet-species {{
                        font-size: 1.2em;
                        opacity: 0.9;
                        margin: 10px 0 0 0;
                    }}
                    .content {{
                        padding: 30px;
                    }}
                    .info-row {{
                        display: flex;
                        justify-content: space-between;
                        margin-bottom: 15px;
                        padding-bottom: 15px;
                        border-bottom: 1px solid #eee;
                    }}
                    .info-row:last-child {{
                        border-bottom: none;
                        margin-bottom: 0;
                    }}
                    .label {{
                        font-weight: bold;
                        color: #555;
                        flex: 1;
                    }}
                    .value {{
                        color: #333;
                        flex: 2;
                        text-align: right;
                    }}
                    .owner-section {{
                        background: #f8f9fa;
                        padding: 20px;
                        margin-top: 20px;
                        border-radius: 10px;
                    }}
                    .owner-title {{
                        font-size: 1.3em;
                        font-weight: bold;
                        color: #4CAF50;
                        margin-bottom: 15px;
                        text-align: center;
                    }}
                    .footer {{
                        text-align: center;
                        padding: 20px;
                        background: #f8f9fa;
                        color: #666;
                        font-size: 0.9em;
                    }}
                    .emergency {{
                        background: #ffe6e6;
                        border-left: 4px solid #ff4444;
                        padding: 15px;
                        margin: 20px 0;
                        border-radius: 5px;
                    }}
                    .emergency-title {{
                        color: #cc0000;
                        font-weight: bold;
                        margin-bottom: 10px;
                    }}
                </style>
            </head>
            <body>
                <div class="card">
                    <div class="header">
                        <img src="{photo_url}" alt="{name}" class="pet-photo">
                        <h1 class="pet-name">{name}</h1>
                        <p class="pet-species">{species} • {breed or 'Mixed Breed'}</p>
                    </div>
                    
                    <div class="content">
                        <div class="info-row">
                            <span class="label">🎂 Date of Birth:</span>
                            <span class="value">{dob_formatted}</span>
                        </div>
                        <div class="info-row">
                            <span class="label">🏷️ Pet ID:</span>
                            <span class="value">#{pet_id}</span>
                        </div>
                        <div class="info-row">
                            <span class="label">📅 Registered:</span>
                            <span class="value">{created_formatted}</span>
                        </div>
                        
                        <div class="owner-section">
                            <div class="owner-title">👤 Owner Information</div>
                            <div class="info-row">
                                <span class="label">Name:</span>
                                <span class="value">{owner_name}</span>
                            </div>
                            <div class="info-row">
                                <span class="label">ID:</span>
                                <span class="value">{owner_cin}</span>
                            </div>
                        </div>
                        
                        <div class="emergency">
                            <div class="emergency-title">🚨 Emergency Information</div>
                            <p>If you found this pet, please contact the Pet Tracker service or the local authorities. This pet is registered and loved!</p>
                        </div>
                    </div>
                    
                    <div class="footer">
                        <p>🐾 Pet Tracker System • Scan NFC for instant access</p>
                        <p>Generated on {datetime.datetime.now().strftime('%Y-%m-%d at %H:%M')}</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Set cache-control headers to prevent constant refreshing
            headers = {'Cache-Control': 'max-age=300'}  # Cache for 5 minutes
            return html_content, 200, headers

    except Exception as e:
        return f"""
        <html>
        <body>
            <h1>Error</h1>
            <p>An error occurred: {str(e)}</p>
        </body>
        </html>
        """, 500
    finally:
        if conn:
            conn.close()

# Simple route for testing animal card access - fixed with proper routing
@app.route('/card/<int:animal_id>')
def simple_animal_card(animal_id):
    """Simple route to access animal card by ID - serves the animal card directly"""
    # Just directly return the animal card - it already has cache headers
    return animal_card(animal_id)

@app.route('/user/profile/<int:user_id>')
def user_profile(user_id):
    """Display user profile with all pets and user information."""
    conn = get_db_connection()
    if conn is None:
        return f"<html><body><h1>Database connection failed</h1></body></html>", 500

    try:
        with conn.cursor() as cur:
            # Get user information
            cur.execute("""
                SELECT u.id, u.cin, u.full_name, u.created_at,
                       COUNT(a.id) as pet_count
                FROM users u 
                LEFT JOIN animals a ON u.id = a.user_id
                WHERE u.id = %s
                GROUP BY u.id, u.cin, u.full_name, u.created_at;
            """, (user_id,))
            
            user_data = cur.fetchone()
            
            if not user_data:
                return f"""
                <html>
                <head>
                    <title>User Not Found</title>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <style>
                        body {{ font-family: Arial, sans-serif; text-align: center; padding: 50px; background-color: #f8f9fa; }}
                        .container {{ max-width: 600px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                        .error {{ color: #e74c3c; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1 class="error">User Not Found</h1>
                        <p>The user with ID {user_id} was not found in our database.</p>
                        <p><a href="javascript:history.back()">Go Back</a></p>
                    </div>
                </body>
                </html>
                """, 404
                
            # Extract user data
            uid, cin, full_name, created_at, pet_count = user_data
            
            # Get all pets of the user
            cur.execute("""
                SELECT id, name, species, breed, date_of_birth, photo_url, created_at
                FROM animals
                WHERE user_id = %s
                ORDER BY created_at DESC;
            """, (user_id,))
            
            pets = cur.fetchall()
            
            # Generate pet cards HTML
            pet_cards_html = ""
            
            for pet in pets:
                pet_id, pet_name, species, breed, dob, photo_url, pet_created_at = pet
                
                # Format date of birth
                dob_formatted = dob.strftime('%Y-%m-%d') if dob else 'Not specified'
                pet_created_formatted = pet_created_at.strftime('%Y-%m-%d') if pet_created_at else 'Not specified'
                
                # Default image if no photo
                # Add cache-busting parameter to photo URL
                if photo_url:
                    photo_url = f"{photo_url}?t={int(datetime.datetime.now().timestamp())}"
                else:
                    # Simple SVG data URL
                    photo_url = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTUwIiBoZWlnaHQ9IjE1MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTUwIiBoZWlnaHQ9IjE1MCIgZmlsbD0iI2NjY2NjYyIvPjx0ZXh0IHg9IjUwJSIgeT0iNTAlIiBmb250LWZhbWlseT0iQXJpYWwiIGZvbnQtc2l6ZT0iMTYiIGZpbGw9IiM2NjY2NjYiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGFsaWdubWVudC1iYXNlbGluZT0ibWlkZGxlIj5ObyBQaG90bzwvdGV4dD48L3N2Zz4='
                
                pet_cards_html += f"""
                <div class="pet-card">
                    <div class="pet-image">
                        <img src="{photo_url}" alt="{pet_name}" onerror="this.src='/uploads/pets/placeholder.svg'">
                    </div>
                    <div class="pet-info">
                        <h3>{pet_name}</h3>
                        <p><strong>Species:</strong> {species or 'Not specified'}</p>
                        <p><strong>Breed:</strong> {breed or 'Not specified'}</p>
                        <p><strong>Born:</strong> {dob_formatted}</p>
                        <p><strong>Added:</strong> {pet_created_formatted}</p>
                        <div class="pet-actions">
                            <a href="/animal/{pet_id}" class="action-button view-button">View Card</a>
                            <a href="/pet/{pet_id}" class="action-button edit-button">Pet Details</a>
                        </div>
                    </div>
                </div>
                """
            
            # Format user created date
            created_formatted = created_at.strftime('%Y-%m-%d %H:%M') if created_at else 'Not specified'
            
            # Generate HTML page
            html_content = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>User Profile - {full_name}</title>
                <style>
                    body {{
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        margin: 0;
                        padding: 0;
                        background-color: #f0f2f5;
                        color: #333;
                    }}
                    
                    .container {{
                        max-width: 1000px;
                        margin: 0 auto;
                        padding: 20px;
                    }}
                    
                    .header {{
                        background: linear-gradient(45deg, #4CAF50, #45a049);
                        color: white;
                        padding: 30px 0;
                        text-align: center;
                        border-radius: 10px;
                        margin-bottom: 30px;
                        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
                    }}
                    
                    .header h1 {{
                        margin: 0;
                        font-size: 2.5em;
                    }}
                    
                    .header p {{
                        margin: 5px 0 0 0;
                        opacity: 0.9;
                    }}
                    
                    .user-info {{
                        background-color: white;
                        border-radius: 10px;
                        padding: 20px;
                        margin-bottom: 30px;
                        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                    }}
                    
                    .user-info h2 {{
                        color: #4CAF50;
                        border-bottom: 2px solid #e9e9e9;
                        padding-bottom: 10px;
                        margin-top: 0;
                    }}
                    
                    .info-grid {{
                        display: grid;
                        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                        gap: 20px;
                    }}
                    
                    .info-item {{
                        background-color: #f9f9f9;
                        padding: 15px;
                        border-radius: 8px;
                    }}
                    
                    .info-item h3 {{
                        margin-top: 0;
                        color: #555;
                        font-size: 0.9em;
                        text-transform: uppercase;
                    }}
                    
                    .info-item p {{
                        font-size: 1.2em;
                        font-weight: bold;
                        margin: 5px 0 0 0;
                    }}
                    
                    .pets-section {{
                        background-color: white;
                        border-radius: 10px;
                        padding: 20px;
                        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                    }}
                    
                    .pets-section h2 {{
                        color: #4CAF50;
                        border-bottom: 2px solid #e9e9e9;
                        padding-bottom: 10px;
                        margin-top: 0;
                    }}
                    
                    .pet-cards {{
                        display: grid;
                        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                        gap: 20px;
                    }}
                    
                    .pet-card {{
                        background-color: #fff;
                        border-radius: 10px;
                        overflow: hidden;
                        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1);
                        transition: transform 0.3s;
                        display: flex;
                    }}
                    
                    .pet-card:hover {{
                        transform: translateY(-5px);
                    }}
                    
                    .pet-image {{
                        width: 150px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        background-color: #f0f0f0;
                    }}
                    
                    .pet-image img {{
                        width: 100%;
                        height: 150px;
                        object-fit: cover;
                    }}
                    
                    .pet-info {{
                        padding: 15px;
                        flex: 1;
                    }}
                    
                    .pet-info h3 {{
                        margin: 0 0 10px 0;
                        color: #4CAF50;
                    }}
                    
                    .pet-info p {{
                        margin: 5px 0;
                        font-size: 0.9em;
                    }}
                    
                    .pet-actions {{
                        margin-top: 15px;
                        display: flex;
                        gap: 10px;
                    }}
                    
                    .action-button {{
                        display: inline-block;
                        padding: 8px 12px;
                        border-radius: 5px;
                        text-decoration: none;
                        font-size: 0.8em;
                        font-weight: bold;
                        text-align: center;
                        transition: background-color 0.3s;
                    }}
                    
                    .view-button {{
                        background-color: #3498db;
                        color: white;
                    }}
                    
                    .view-button:hover {{
                        background-color: #2980b9;
                    }}
                    
                    .edit-button {{
                        background-color: #f39c12;
                        color: white;
                    }}
                    
                    .edit-button:hover {{
                        background-color: #e67e22;
                    }}
                    
                    .footer {{
                        text-align: center;
                        margin-top: 30px;
                        padding: 20px;
                        color: #777;
                        font-size: 0.9em;
                    }}
                    
                    @media (max-width: 768px) {{
                        .pet-cards {{
                            grid-template-columns: 1fr;
                        }}
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>{full_name}</h1>
                        <p>Pet Owner Profile</p>
                    </div>
                    
                    <div class="user-info">
                        <h2>User Information</h2>
                        <div class="info-grid">
                            <div class="info-item">
                                <h3>ID Number</h3>
                                <p>{cin}</p>
                            </div>
                            <div class="info-item">
                                <h3>Account Created</h3>
                                <p>{created_formatted}</p>
                            </div>
                            <div class="info-item">
                                <h3>Total Pets</h3>
                                <p>{pet_count}</p>
                            </div>
                            <div class="info-item">
                                <h3>User ID</h3>
                                <p>#{uid}</p>
                            </div>
                        </div>
                    </div>
                    
                    <div class="pets-section">
                        <h2>Pet Collection ({pet_count})</h2>
                        <div class="pet-cards">
                            {pet_cards_html if pet_count > 0 else "<p>No pets registered yet.</p>"}
                        </div>
                    </div>
                    
                    <div class="footer">
                        <p>🐾 Pet Tracker System • {datetime.datetime.now().strftime('%Y-%m-%d at %H:%M')}</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            return html_content

    except Exception as e:
        return f"""
        <html>
        <body>
            <h1>Error</h1>
            <p>An error occurred: {str(e)}</p>
        </body>
        </html>
        """, 500
    finally:
        if conn:
            conn.close()

@app.route('/user/<int:user_id>/pet-locations', methods=['GET'])
def user_pet_locations(user_id):
    """Returns the latest location for each pet of the user as JSON."""
    conn = get_db_connection()
    if conn is None:
        return jsonify({'status': 'error', 'message': 'Database connection failed.'}), 500
    try:
        with conn.cursor() as cur:
            # Get all pets for the user
            cur.execute("SELECT id, name FROM animals WHERE user_id = %s;", (user_id,))
            pets = cur.fetchall()
            results = []
            for pet_id, name in pets:
                cur.execute(
                    "SELECT latitude, longitude FROM locations WHERE pet_id = %s ORDER BY timestamp DESC LIMIT 1",
                    (pet_id,)
                )
                loc = cur.fetchone()
                if loc:
                    results.append({
                        'pet_id': pet_id,
                        'name': name,
                        'latitude': float(loc[0]),
                        'longitude': float(loc[1])
                    })
            return jsonify({'status': 'success', 'locations': results})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        if conn:
            conn.close()

@app.route('/pet/<int:pet_id>/photo', methods=['POST'])
def upload_pet_photo(pet_id):
    """Upload a photo for a specific pet."""
    print(f"Photo upload called for pet_id: {pet_id}")
    # No authentication required for this endpoint
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part in the request.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected.'}), 400

    conn = get_db_connection()
    if conn is None:
        return jsonify({'status': 'error', 'message': 'Database connection failed'}), 500

    try:
        with conn.cursor() as cur:
            # Just check if the pet exists
            cur.execute("SELECT id FROM animals WHERE id = %s;", (pet_id,))
            pet = cur.fetchone()
            if not pet:
                return jsonify({'status': 'error', 'message': 'Pet not found'}), 404

        # Create uploads directory if it doesn't exist
        upload_dir = os.path.join(os.getcwd(), 'uploads', 'pets')
        os.makedirs(upload_dir, exist_ok=True)
        
        # Generate unique filename
        file_extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'jpg'
        filename = f"pet_{pet_id}_{int(datetime.datetime.now().timestamp())}.{file_extension}"
        file_path = os.path.join(upload_dir, filename)
        
        # Save file
        file.save(file_path)
        
        # Generate URL (assuming the server serves static files from /uploads)
        # Store as a simple string path that can be used as a URL
        photo_url = f"/uploads/pets/{filename}"
        
        # Update database - photo_url is stored as varchar in DB
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE animals SET photo_url = %s WHERE id = %s RETURNING id, name;",
                (photo_url, pet_id)
            )
            updated_pet = cur.fetchone()
            conn.commit()

            if updated_pet:
                return jsonify({
                    'status': 'success',
                    'message': 'Photo uploaded successfully',
                    'photo_url': photo_url,
                    'pet': {'id': updated_pet[0], 'name': updated_pet[1]}
                })
            else:
                return jsonify({'status': 'error', 'message': 'Failed to update pet photo URL.'}), 500

    except Exception as e:
        conn.rollback()
        return jsonify({'status': 'error', 'message': f'Upload error: {str(e)}'}), 500
    finally:
        if conn:
            conn.close()

# Add route to serve uploaded files
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    uploads_dir = os.path.join(os.getcwd(), 'uploads')
    return send_from_directory(uploads_dir, filename)

if __name__ == '__main__':
    app.run(port=5002, debug=False, use_reloader=False)
