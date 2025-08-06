# Drupal Data Fetcher

An overview of the Drupal community through its users, projects and issues.

## About the Author

**Matthieu Scarset** is a seasoned Senior Full Stack Developer and an emerging Data Scientist. Building Drupal applications since 2014, he has successfully helped dozens of organizations deploy effective solutions and elevate customer experience. Beyond his client work, Matthieu is regarded in the Drupal community for his significant contributions to core issues and widely used modules such as _Menu Manipulator_ and _Calendar Views_. 

Connect with Matthieu on LinkedIn: [linkedin.com/in/matthieuscarset](https://linkedin.com/in/matthieuscarset)

## Getting started

### GCP Service Account

```bash
SA=$SERVICE_ACCOUNT@$GCP_PROJECT_ID.iam.gserviceaccount.com
gcloud iam service-accounts create $SERVICE_ACCOUNT --display-name $DISPLAY_NAME
gcloud iam service-accounts keys create ~/gcp/$SERVICE_ACCOUNT.json --iam-account $SA
gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
    --member="serviceAccount:$SA" \
    --role=$ROLE_OWNER
```

## References

...

## Credits

Special thanks to the amazing people building and maintaining the Drupal.org website and for making public such a robust REST API. 

- **Neil Drumm** ([drumm](https://www.drupal.org/u/drumm)) for unblocking me when my bot was disrespectful of the request rate 😅
- **Fran Garcia-Linares** ([fjgarlin](https://www.drupal.org/u/fjgarlin)) for his help when I was learning more about the limitations of the current REST API

