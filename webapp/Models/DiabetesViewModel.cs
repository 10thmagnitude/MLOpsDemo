using System;
using System.ComponentModel.DataAnnotations;

namespace webapp.Models
{
    public class DiabetesViewModel
    {
        [Required]
        [Display(Name = "Number of Pregnancies")]
        public int NumPregnancies { get; set; }

        [Required]
        [Display(Name = "Glucose (mmol/L)")]
        public int Glucose { get; set; }

        [Required]
        [Display(Name = "Diastolic BP (mm Hg)")]
        public int Diastolic { get; set; }

        [Required]
        [Display(Name = "Triceps skin fold thickness (mm)")]
        public double Thickness { get; set; }

        [Required]
        [Display(Name = "Insulin (mu U/ml)")]
        public double Insulin { get; set; }

        [Required]
        [Display(Name = "BMI (kg/m^2)")]
        public double BMI { get; set; }

        [Required]
        [Display(Name = "Diabetes pedigree function")]
        public double Pedigree { get; set; }

        [Required]
        [Display(Name = "Age")]
        public int Age { get; set; }

        [Display(Name = "Diabetes")]
        public bool? Diabetes { get; set; } 
    }
}