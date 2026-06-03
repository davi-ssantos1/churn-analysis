from pydantic import BaseModel, Field, field_validator


class CustomerRecord(BaseModel):
    """
    Make a custom contract to data load the raw .csv file to a processed .db SQL file
    """

    customer_id: str = Field(alias="customerID")
    gender: str
    senior_citizen: bool = Field(alias="SeniorCitizen")
    partner: bool = Field(alias="Partner")
    dependents: bool = Field(alias="Dependents")
    tenure: int = Field(ge=0)
    phone_service: bool = Field(alias="PhoneService")
    multiple_lines: str = Field(alias="MultipleLines")
    internet_services: str = Field(alias="InternetService")
    online_security: bool = Field(alias="OnlineSecurity")
    online_backup: bool = Field(alias="OnlineBackup")
    device_protection: bool = Field(alias="DeviceProtection")
    tech_support: bool = Field(alias="TechSupport")
    streaming_tv: bool = Field(alias="StreamingTV")
    streaming_movies: bool = Field(alias="StreamingMovies")
    contract: str = Field(alias="Contract")
    paper_less_billing: bool = Field(alias="PaperlessBilling")
    payment_method: str = Field(alias="PaymentMethod")
    monthly_charges: float = Field(alias="MonthlyCharges", ge=0)
    total_charges: float | None = Field(alias="TotalCharges", ge=0)
    churn: bool = Field(alias="Churn")

    @field_validator("senior_citizen", mode="before")
    @classmethod
    def transform_0_1_to_true_false(cls, value: bool | int) -> bool:
        """
        Transform 0s and 1s integer features into True and False.
        """
        if isinstance(value, bool):
            return value

        if value == 0:
            return False
        elif value == 1:
            return True
        else:
            raise ValueError(f"Value must to be 0 or 1.\nValue: {value}")

    @field_validator(
        "partner",
        "dependents",
        "phone_service",
        "paper_less_billing",
        "churn",
        mode="before",
    )
    @classmethod
    def transform_yes_no_to_true_false(cls, value: str | bool) -> bool:
        """
        Transform 'Yes' or 'No' string features into True or False.
        """
        if isinstance(value, bool):
            return value

        if isinstance(value, str):
            strip_value = value.strip().lower()
            if strip_value == "yes":
                return True
            elif strip_value == "no":
                return False
            else:
                raise ValueError(f"Value has to be 'yes' or 'no': {value}")
        raise TypeError("Value must be a string or boolean.")

    @field_validator(
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
        mode="before",
    )
    @classmethod
    def transform_yes_no_notinternetservice_to_true_false(
        cls, value: str | bool
    ) -> bool:
        """
        Transform 'Yes', 'No' or 'Not Internet Service' string features to True or False.
        """
        if isinstance(value, bool):
            return value

        if isinstance(value, str):
            strip_value = value.strip().lower()
            if strip_value == "yes":
                return True
            elif strip_value in ("no", "no internet service"):
                return False
            else:
                raise ValueError(
                    f"Value has to be 'yes', 'no' or 'no internet service'.\nValue: {value}"
                )
        raise TypeError("Value must be a string or boolean")

    @field_validator("total_charges", mode="before")
    @classmethod
    def transform_empty_values_to_none(cls, value: str | float | None) -> float | None:
        if value is None or isinstance(value, float):
            return value

        if isinstance(value, str):
            strip_value = value.strip()
            if strip_value == "":
                return None
            try:
                return float(strip_value)
            except ValueError:
                raise ValueError(f"Cannot convert {value} to float.") from None

        raise TypeError(f"{value} must to be a string, float or None.")
