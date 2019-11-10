using System;
using System.ComponentModel.DataAnnotations;
using System.Text.Json.Serialization;

namespace webapp.Models
{
    public class DiabetesViewModel
    {
        [Required]
        [Display(Name = "Number of Pregnancies")]
        [JsonPropertyName("numpreg")]
        public int NumPregnancies { get; set; }

        [Required]
        [Display(Name = "Glucose (mmol/L)")]
        [JsonPropertyName("glucose_conc")]
        public int Glucose { get; set; }

        [Required]
        [Display(Name = "Diastolic BP (mm Hg)")]
        [JsonPropertyName("diastolic_bp")]
        public int Diastolic { get; set; }

        [Required]
        [Display(Name = "Triceps skin fold thickness (mm)")]
        [JsonPropertyName("thickness")]
        public double Thickness { get; set; }

        [Required]
        [Display(Name = "Insulin (mu U/ml)")]
        [JsonPropertyName("insulin")]
        public double Insulin { get; set; }

        [Required]
        [Display(Name = "BMI (kg/m^2)")]
        [JsonPropertyName("bmi")]
        public double BMI { get; set; }

        [Required]
        [Display(Name = "Diabetes pedigree function")]
        [JsonPropertyName("diab_pred")]
        public double Pedigree { get; set; }

        [Required]
        [Display(Name = "Age")]
        [JsonPropertyName("age")]
        public int Age { get; set; }
    }
}