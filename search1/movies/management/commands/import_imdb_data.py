import csv
from django.core.management.base import BaseCommand
from movies.models import Movie

class Command(BaseCommand):
    help = 'Imports movie data from cleaned_imdb_data.csv in the project root'

    def handle(self, *args, **kwargs):
        csv_path = '/app/cleaned_imdb_data.csv'  # Path in Docker container
        try:
            with open(csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                expected_columns = [
                    'Poster_Link', 'Series_Title', 'Released_Year', 'Certificate', 'Runtime',
                    'Genre', 'IMDB_Rating', 'Overview', 'Meta_score', 'Director',
                    'Star1', 'Star2', 'Star3', 'Star4', 'No_of_Votes', 'Gross'
                ]
                if not all(col in reader.fieldnames for col in expected_columns):
                    missing = [col for col in expected_columns if col not in reader.fieldnames]
                    raise KeyError(f"Missing columns: {missing}")

                for row in reader:
                    try:
                        # Handle Meta_score: Convert float to int, handle empty or invalid
                        meta_score = None
                        if row['Meta_score'] and row['Meta_score'].strip() and row['Meta_score'] != '0':
                            try:
                                meta_score = int(float(row['Meta_score']))
                            except ValueError:
                                self.stdout.write(self.style.WARNING(f"Invalid Meta_score '{row['Meta_score']}' for {row['Series_Title']}"))

                        # Handle Gross: Remove decimals, handle empty
                        gross = ''
                        if row['Gross'] and row['Gross'].strip() and row['Gross'] != '0':
                            try:
                                gross = str(int(float(row['Gross'])))
                            except ValueError:
                                self.stdout.write(self.style.WARNING(f"Invalid Gross '{row['Gross']}' for {row['Series_Title']}"))

                        # Handle Released_Year: Convert to int, handle empty or invalid
                        released_year = None
                        if row['Released_Year'] and row['Released_Year'].strip():
                            try:
                                released_year = int(row['Released_Year'])
                            except ValueError:
                                self.stdout.write(self.style.WARNING(f"Invalid Released_Year '{row['Released_Year']}' for {row['Series_Title']}"))

                        Movie.objects.update_or_create(
                            series_title=row['Series_Title'],
                            defaults={
                                'released_year': released_year,
                                'runtime': row['Runtime'] or '',
                                'genre': row['Genre'] or '',
                                'rating': float(row['IMDB_Rating']) if row['IMDB_Rating'] and row['IMDB_Rating'].strip() else None,
                                'no_of_votes': int(row['No_of_Votes']) if row['No_of_Votes'] and row['No_of_Votes'].strip() else 0,
                                'director': row['Director'] or '',
                                'star1': row['Star1'] or '',
                                'star2': row['Star2'] or '',
                                'star3': row['Star3'] or '',
                                'star4': row['Star4'] or '',
                                'overview': row['Overview'] or '',
                                'poster_link': row['Poster_Link'] or '',
                                'awards': '',
                                'certificate': row['Certificate'] or '',
                                'gross': gross,
                                'meta_score': meta_score,
                                'embedding': None,
                            }
                        )
                        self.stdout.write(self.style.SUCCESS(f"Imported: {row['Series_Title']}"))
                    except Exception as e:
                        self.stderr.write(self.style.ERROR(f"Error importing {row['Series_Title']}: {str(e)}"))
                self.stdout.write(self.style.SUCCESS('Movie data imported successfully!'))
        except FileNotFoundError:
            self.stderr.write(self.style.ERROR(f"CSV file not found at {csv_path}"))
        except KeyError as e:
            self.stderr.write(self.style.ERROR(f"Column error: {str(e)}"))
        except Exception as e:
            self.stderr.write(self.style.ERROR(f"General error: {str(e)}"))