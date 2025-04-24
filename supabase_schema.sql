-- Step 1: Create the table
create table if not exists public.food_logs (
  id int8 primary key generated always as identity,
  timestamp timestamptz default now(),
  meal_time text,
  description text,
  items text,
  calories text,
  total_calories float8,
  health_score float8,
  rationale text,
  macronutrient_estimate text,
  macros_protein float8,
  macros_fat float8,
  macros_carb float8,
  eat_frequency text,
  ideal_comparison text,
  mood_impact text,
  summary text,
  image_url text
);

-- Step 2: Enable RLS
alter table public.food_logs enable row level security;

-- Step 3: Add default "Allow all" policy for now (MVP-friendly)
create policy "Allow all actions for now"
on public.food_logs
for all
using (true);